#!/usr/bin/env python

#
# Copyright (c) 2022 German BioImaging
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import contextlib
import gzip
import sys
import json
import logging
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Set, Tuple, Union

from importlib.metadata import entry_points
from omero.cli import BaseControl, Parser, ProxyStringType
from omero.gateway import BlitzGateway, BlitzObjectWrapper
from omero.model import Dataset, Image, IObject, Plate, Project, Screen
from omero.sys import ParametersI
from omero_marshal import get_encoder
from pyld import jsonld
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, Namespace as NS, RDF, RDFS, SDO
from rdflib_pyld_compat import pyld_jsonld_from_rdflib_graph

HELP = """A plugin for exporting RDF from OMERO

omero-rdf creates a stream of RDF triples from the starting object that
it is given. This may be one of: Image, Dataset, Project, Plate, and Screen.

Examples:

  omero rdf Image:123                # Streams each triple found in N-Triples format

  omero rdf -F=jsonld Image:123      # Collects all triples and prints formatted output
  omero rdf -S=flat Project:123      # Do not recurse into containers ("flat-strategy")
  omero rdf -S=sample Project:123    # Get only the first element from each containers ("sample-strategy")
  omero rdf --trim-whitespace ...    # Strip leading and trailing whitespace from text
  omero rdf --first-handler-wins ... # First mapping wins; others will be ignored

  omero rdf --file - ...             # Write RDF triples to stdout
  omero rdf --file output.nt ...     # Write RDF triples to the specified file
  omero rdf --file output.nt.gz      # Write RDF triples to the specified file, gzipping

"""

# TYPE DEFINITIONS

Data = Dict[str, Any]
Subj = Union[BNode, URIRef]
Obj = Union[BNode, Literal, URIRef]
Triple = Tuple[Subj, URIRef, Obj]
Handlers = List[Callable[[URIRef, URIRef, Data], Generator[Triple, None, bool]]]

# NAMESPACE DEFINITIONS

NS_OME = NS("http://www.openmicroscopy.org/Schemas/OME/2016-06#")
NS_OMERO = NS("http://www.openmicroscopy.org/Schemas/OMERO/2016-06#")

STANDARD_ANNOTATIONS = {"Organism": NS_OME["Organism"]}


@contextlib.contextmanager
def open_with_default(filename=None, filehandle=None):
    """
    Open a file for writing if given and close on completion.

    No closing will happen if the file name is "-" since stdout will be used.
    If no filehandle is given, stdout will also be used.
    Otherwise return the given filehandle will be used.
    """
    close = False
    if filename:
        if filename == "-":
            fh = sys.stdout
        else:
            if filename.endswith(".gz"):
                fh = gzip.open(filename, "wt")
            else:
                fh = open(filename, "w")
            close = True
    else:
        if filehandle is None:
            filehandle = sys.stdout
        fh = filehandle

    try:
        yield fh
    finally:
        if close:
            fh.close()


def gateway_required(func: Callable) -> Callable:  # type: ignore
    """
    Decorator which initializes a client (self.client),
    a BlitzGateway (self.gateway), and makes sure that
    all services of the Blitzgateway are closed again.

    FIXME: copied from omero-cli-render. move upstream
    """

    @wraps(func)
    def _wrapper(self, *args: Any, **kwargs: Any):  # type: ignore
        self.client = self.ctx.conn(*args)
        self.gateway = BlitzGateway(client_obj=self.client)

        try:
            return func(self, *args, **kwargs)
        finally:
            if self.gateway is not None:
                self.gateway.close(hard=False)
                self.gateway = None
                self.client = None

    return _wrapper


class Format:
    """
    Output mechanisms split into two types: streaming and non-streaming.
    Critical methods include:

        - streaming:
            - serialize_triple: return a representation of the triple
        - non-streaming:
            - add: store a triple for later serialization
            - serialize_graph: return a representation of the graph

    See the subclasses for more information.
    """

    def __init__(self):
        self.streaming = None

    def __str__(self):
        return self.__class__.__name__[:-6].lower()

    def __lt__(self, other):
        return str(self) < str(other)

    def add(self, triple):
        raise NotImplementedError()

    def serialize_triple(self, triple):
        raise NotImplementedError()

    def serialize_graph(self):
        raise NotImplementedError()


class StreamingFormat(Format):
    def __init__(self):
        super().__init__()
        self.streaming = True

    def add(self, triple):
        raise RuntimeError("adding not supported during streaming")

    def serialize_graph(self):
        raise RuntimeError("graph serialization not supported during streaming")


class NTriplesFormat(StreamingFormat):
    def __init__(self):
        super().__init__()

    def serialize_triple(self, triple):
        graph = Graph()
        graph.add(triple)
        return graph.serialize(format="nt11")


class NonStreamingFormat(Format):
    def __init__(self):
        super().__init__()
        self.streaming = False
        self.graph = Graph()
        self.graph.bind("", NS_OME)
        self.graph.bind("omero", NS_OMERO)
        self.graph.bind("dcterms", DCTERMS)
        # TODO: Allow handlers to register namespaces

    def add(self, triple):
        self.graph.add(triple)

    def serialize_triple(self, triple):
        raise RuntimeError("triple serialization not supported during streaming")


class TurtleFormat(NonStreamingFormat):
    def __init__(self):
        super().__init__()

    def serialize_graph(self) -> None:
        return self.graph.serialize()


class JSONLDFormat(NonStreamingFormat):
    def __init__(self):
        super().__init__()

    def context(self):
        # TODO: allow handlers to add to this
        return {
            "@vocab": str(NS_OME),
            "omero": str(NS_OMERO),
            "dcterms": str(DCTERMS),
        }

    def serialize_graph(self) -> None:
        return self.graph.serialize(
            format="json-ld",
            context=self.context(),
            indent=4,
        )


class ROCrateFormat(JSONLDFormat):
    def __init__(self):
        super().__init__()

    def context(self):
        ctx = super().context()
        ctx["@rocrate"] = "https://w3id.org/ro/crate/1.1/context"
        return ctx

    def serialize_graph(self):
        ctx = self.context()
        j = pyld_jsonld_from_rdflib_graph(self.graph)
        j = jsonld.flatten(j, ctx)
        j = jsonld.compact(j, ctx)
        if "@graph" not in j:
            raise Exception(j)
        j["@graph"][0:0] = [
            {
                "@id": "./",
                "@type": "Dataset",
                "rocrate:license": "https://creativecommons.org/licenses/by/4.0/",
            },
            {
                "@id": "ro-crate-metadata.json",
                "@type": "CreativeWork",
                "rocrate:conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                "rocrate:about": {"@id": "./"},
            },
        ]
        return json.dumps(j, indent=4)


def format_mapping():
    return {
        "ntriples": NTriplesFormat(),
        "jsonld": JSONLDFormat(),
        "turtle": TurtleFormat(),
        "ro-crate": ROCrateFormat(),
    }


def format_list():
    return format_mapping().keys()


class Handler:
    """
    Instances are used to generate triples.

    Methods which can be subclassed:
        TBD

    """

    def __init__(
        self,
        gateway: BlitzGateway,
        formatter: Format,
        trim_whitespace=False,
        use_ellide=False,
        first_handler_wins=False,
        descent="recursive",
        filehandle=sys.stdout,
    ) -> None:
        self.gateway = gateway
        self.cache: Set[URIRef] = set()
        self.bnode = 0
        self.formatter = formatter
        self.trim_whitespace = trim_whitespace
        self.use_ellide = use_ellide
        self.first_handler_wins = first_handler_wins
        self.descent = descent
        self._descent_level = 0
        # Annotation handlers may be changed in adaptations of the plugin
        self.annotation_handlers = self.load_handlers()
        self.info = self.load_server()
        self.filehandle = filehandle

    def skip_descent(self):
        return self.descent != "recursive" and self._descent_level > 0

    def get_sample(self):
        return self.descent == "sample"

    def descending(self):
        self._descent_level += 1

    # This delegates the handling of annotations to configurations on particular repositories
    # It is done, for example, on https://github.com/German-BioImaging/omero-rdf-wikidata
    def load_handlers(self) -> Handlers:
        annotation_handlers: Handlers = []

        # Adds room for using diverse annotation_handlers
        # The default pipeline is used if they are not present
        eps = entry_points()
        for ep in eps.get("omero_rdf.annotation_handler", []):
            ah_loader = ep.load()
            annotation_handlers.append(ah_loader(self))
        return annotation_handlers

    def load_server(self) -> Any:
        # Attempt to auto-detect server
        comm = self.gateway.c.getCommunicator()
        return self.gateway.c.getRouter(comm).ice_getEndpoints()[0].getInfo()

    def get_identity(self, _type: str, _id: Any) -> URIRef:

        # Workaround for object names ending with I from the gateway, e.g. ImageI
        if _type.endswith("I") and _type != ("ROI"):
            _type = _type[0:-1]
        return URIRef(f"https://{self.info.host}/{_type}/{_id}")

    def get_bnode(self) -> BNode:
        try:
            return BNode()
            # return f":b{self.bnode}"
        finally:
            self.bnode += 1

    def get_type(self, data: Data) -> str:
        return data.get("@type", "UNKNOWN").split("#")[-1]

    def literal(self, v: Any) -> Literal:
        """
        Prepare Python objects for use as literals
        """
        if isinstance(v, str):
            v = str(v)
            if self.use_ellide and len(v) > 50:
                v = f"{v[0:24]}...{v[-20:-1]}"
            elif v.startswith(" ") or v.endswith(" "):
                if self.trim_whitespace:
                    v = v.strip()
                else:
                    logging.warning(
                        "string has whitespace that needs trimming: '%s'", v
                    )
        return Literal(v)

    def get_class(self, o):
        if isinstance(o, IObject):
            c = o.__class__
        else:  # Wrapper
            c = o._obj.__class__
        return c

    def __call__(self, o: BlitzObjectWrapper) -> URIRef:
        c = self.get_class(o)
        encoder = get_encoder(c)
        if encoder is None:
            raise Exception(f"unknown: {c}")
        else:
            # TODO: could potentially do this once for the top object
            data = encoder.encode(o, include_context=False)
            return self.handle(data)

    def annotations(self, obj, objid):
        """
        Loop through all annotations and handle them individually.
        """
        if isinstance(obj, IObject):
            # Not a wrapper object
            for annotation in obj.linkedAnnotationList():

                # Calls the functor instance of Handler
                # Which parsers annotation with the omero-marshal encoder
                # E.g. https://github.com/ome/omero-marshal/blob/master/omero_marshal/encode/encoders/map_annotation.py
                annid = self(annotation)
                self.contains(objid, annid)
                if self.get_sample():
                    break
        else:
            for annotation in obj.listAnnotations(None):
                obj._loadAnnotationLinks()

                # Calls the functor instance of Handler
                # Which parsers annotation with the omero-marshal encoder
                # E.g. https://github.com/ome/omero-marshal/blob/master/omero_marshal/encode/encoders/map_annotation.py
                annid = self(annotation)
                self.contains(objid, annid)
                if self.get_sample():
                    break

    def handle(self, data: Data) -> URIRef:
        """
        Parses the data object into RDF triples.

        Returns the id for the data object itself
        """
        # TODO: Add quad representation as an option

        str_id = data.get("@id")
        if not str_id:
            raise Exception(f"missing id: {data}")

        # TODO: this call is likely redundant
        _type = self.get_type(data)
        _id = self.get_identity(_type, str_id)

        for triple in self.rdf(_id, data):

            # Hardcode mapping from ome:Name to rdfs:label and ome:Description to SDO:description
            # This follows the choices on Wikidata.
            if "http://www.openmicroscopy.org/Schemas/OME/2016-06#Name" in triple[1]:
                new_triple = (triple[0], RDFS.label, triple[2])
                triple = new_triple

            if (
                "http://www.openmicroscopy.org/Schemas/OME/2016-06#Description"
                in triple[1]
            ):
                new_triple = (triple[0], SDO.description, triple[2])
                triple = new_triple

            if triple:
                if None in triple:
                    logging.debug("skipping None value: %s %s %s", triple)
                else:
                    self.emit(triple)

        return _id

    def contains(self, parent, child):
        """
        Use emit to generate isPartOf and hasPart triples

        TODO: add an option to only choose one of the two directions.
        """
        self.emit((child, DCTERMS.isPartOf, parent))
        self.emit((parent, DCTERMS.hasPart, child))

    def emit(self, triple: Triple):
        if self.formatter.streaming:
            print(self.formatter.serialize_triple(triple), file=self.filehandle)
        else:
            self.formatter.add(triple)

    def close(self):
        if not self.formatter.streaming:
            print(self.formatter.serialize_graph(), file=self.filehandle)

    def rdf(
        self,
        _id: Subj,
        data: Data,
    ) -> Generator[Triple, None, None]:

        _type = self.get_type(data)

        # Temporary workaround while deciding how to pass annotations
        # Checking length is redundant; just a sanity check, as
        # the list of annotation_handlers can be empty.
        if len(self.annotation_handlers) > 0:
            if "Annotation" in str(_type):
                for ah in self.annotation_handlers:
                    handled = yield from ah(
                        None,
                        None,
                        data,
                    )
                    if self.first_handler_wins and handled:
                        return
        # End workaround

        if _id in self.cache:
            logging.debug("# skipping previously seen %s", _id)
            return
        else:
            self.cache.add(_id)

        for top_level_key, top_level_value in sorted(data.items()):

            if top_level_key == "@type":
                if top_level_value.startswith("http"):
                    yield (_id, RDF.type, URIRef(top_level_value))
                elif top_level_value.startswith("omero:"):
                    yield (_id, RDF.type, NS_OMERO[top_level_value])
                else:
                    yield (_id, RDF.type, NS_OME[top_level_value])

            elif top_level_key in ("@id", "omero:details", "Annotations"):
                # Types that we want to omit for now
                pass
            else:
                # Refactor back to get_key? TODO
                if top_level_key.startswith("omero:"):
                    key_as_uri = NS_OMERO[top_level_key[6:]]
                else:
                    key_as_uri = NS_OME[top_level_key]

                # Check if the top level value is itself an object
                if isinstance(top_level_value, dict):
                    if "@id" in top_level_value:
                        yield from self.yield_object_with_id(
                            _id, key_as_uri, top_level_value
                        )

                    else:
                        # Without an identity, use a bnode
                        # TODO: store by value for re-use?
                        bnode = self.get_bnode()
                        yield (_id, key_as_uri, bnode)
                        yield from self.rdf(bnode, top_level_value)

                # Here below may parse a [[key,value], [[key,value]] construct
                # It is found in some annotation kinds
                elif isinstance(top_level_value, list):

                    for item in top_level_value:
                        # The item is an object and should be handled completely
                        if isinstance(item, dict) and "@id" in item:
                            yield from self.yield_object_with_id(_id, key_as_uri, item)

                        # The object is one nested list with key, value pairs (length of 2)
                        elif isinstance(item, list) and len(item) == 2:

                            annotation_key = item[0]
                            annotation_value = item[1]

                            # Parses expected key-value pairs in a direct way
                            # TODO validate the annotation_value;
                            # parse in direct way only if it is well-formed
                            if _type == "MapAnnotation":
                                if annotation_key in STANDARD_ANNOTATIONS.keys():
                                    annotation_key_uri = STANDARD_ANNOTATIONS[
                                        annotation_key
                                    ]
                                    yield (
                                        _id,
                                        annotation_key_uri,
                                        self.literal(annotation_value),
                                    )
                                # Creates a blank node for other kinds
                                else:
                                    bnode = self.get_bnode()

                                    # TODO: KVPs need ordering info, also no use of "key" here.
                                    yield (_id, NS_OME["Map"], bnode)
                                    yield (
                                        bnode,
                                        NS_OME["Key"],
                                        self.literal(annotation_key),
                                    )
                                    yield (
                                        bnode,
                                        NS_OME["Value"],
                                        self.literal(annotation_value),
                                    )

                        else:
                            raise Exception(f"unknown list item: {item}")
                else:
                    yield (_id, key_as_uri, self.literal(top_level_value))

        # Creates the room for special handling for Annotations
        # If not, default to the base processing from self.rdf()
        # This is present, for example, on some Image objects
        annotations = data.get("Annotations", [])
        for annotation in annotations:
            handled = False
            if len(self.annotation_handlers) > 0:
                for ah in self.annotation_handlers:
                    handled = yield from ah(_id, NS_OME["annotation"], annotation)
                    if handled:
                        break
            if not handled:  # TODO: could move to a default handler
                aid = self.get_identity("AnnotationTBD", annotation["@id"])
                yield (_id, NS_OME["annotation"], aid)
                yield from self.rdf(aid, annotation)

    def yield_object_with_id(self, _id, key, v):
        """
        Yields a link to the object as well as its representation.
        """
        v_type = self.get_type(v)
        val = self.get_identity(v_type, v["@id"])
        yield (_id, key, val)
        yield from self.rdf(_id, v)


class RdfControl(BaseControl):
    def _configure(self, parser: Parser) -> None:
        parser.add_login_arguments()
        rdf_type = ProxyStringType("Image")
        rdf_help = "Object to be exported to RDF"
        parser.add_argument("target", type=rdf_type, nargs="+", help=rdf_help)
        format_group = parser.add_mutually_exclusive_group()
        format_group.add_argument(
            "--pretty",
            action="store_true",
            default=False,
            help="Shortcut for --format=turtle",
        )
        format_group.add_argument(
            "--format",
            "-F",
            default="ntriples",
            choices=format_list(),
        )
        parser.add_argument(
            "--descent",
            "-S",
            default="recursive",
            help="Descent strategy to use: recursive, flat, sample",
        )
        parser.add_argument(
            "--ellide", action="store_true", default=False, help="Shorten strings"
        )
        parser.add_argument(
            "--first-handler-wins",
            "-1",
            action="store_true",
            default=False,
            help="Don't duplicate annotations",
        )
        parser.add_argument(
            "--trim-whitespace",
            action="store_true",
            default=False,
            help="Remove leading and trailing whitespace from literals",
        )
        parser.add_argument(
            "--file",
            type=str,
            default=None,
            help="Write RDF triples to the specified file",
        )
        parser.set_defaults(func=self.action)

    @gateway_required
    def action(self, args: Namespace) -> None:

        # Support hidden --pretty flag
        if args.pretty:
            args.format = TurtleFormat()
        else:
            args.format = format_mapping()[args.format]

        with open_with_default(args.file) as fh:
            handler = Handler(
                self.gateway,
                formatter=args.format,
                use_ellide=args.ellide,
                trim_whitespace=args.trim_whitespace,
                first_handler_wins=args.first_handler_wins,
                descent=args.descent,
                filehandle=fh,
            )
            self.descend(self.gateway, args.target, handler)
            handler.close()

    # TODO: move to handler?
    def descend(
        self,
        gateway: BlitzGateway,
        target: IObject,
        handler: Handler,
    ) -> URIRef:
        """
        Copied from omero-cli-render. Should be moved upstream
        """

        if isinstance(target, list):
            return [self.descend(gateway, t, handler) for t in target]

        # "descent" doesn't apply to a list
        if handler.skip_descent():
            objid = handler(target)
            logging.debug("skip descent: %s", objid)
            return objid
        else:
            handler.descending()

        if isinstance(target, Screen):
            scr = self._lookup(gateway, "Screen", target.id)
            scrid = handler(scr)
            for plate in scr.listChildren():
                pltid = self.descend(gateway, plate._obj, handler)
                handler.contains(scrid, pltid)
                if handler.get_sample():
                    break
            handler.annotations(scr, scrid)
            return scrid

        elif isinstance(target, Plate):
            plt = self._lookup(gateway, "Plate", target.id)
            pltid = handler(plt)
            handler.annotations(plt, pltid)
            for well in plt.listChildren():
                wid = handler(well)  # No descend
                handler.contains(pltid, wid)
                for idx in range(0, well.countWellSample()):
                    img = well.getImage(idx)
                    imgid = self.descend(gateway, img._obj, handler)
                    handler.contains(wid, imgid)
                    if handler.get_sample():
                        break
            return pltid

        elif isinstance(target, Project):
            prj = self._lookup(gateway, "Project", target.id)
            prjid = handler(prj)
            handler.annotations(prj, prjid)
            for ds in prj.listChildren():
                dsid = self.descend(gateway, ds._obj, handler)
                handler.contains(prjid, dsid)
                if handler.get_sample():
                    break
            return prjid

        elif isinstance(target, Dataset):
            ds = self._lookup(gateway, "Dataset", target.id)
            dsid = handler(ds)
            handler.annotations(ds, dsid)
            for img in ds.listChildren():
                imgid = self.descend(gateway, img._obj, handler)
                handler.contains(dsid, imgid)
                if handler.get_sample():
                    break
            return dsid

        elif isinstance(target, Image):
            img = self._lookup(gateway, "Image", target.id)
            imgid = handler(img)
            if img.getPrimaryPixels() is not None:
                pixid = handler(img.getPrimaryPixels())
                handler.contains(imgid, pixid)
            handler.annotations(img, imgid)
            for roi in self._get_rois(gateway, img):
                roiid = handler(roi)
                handler.annotations(roi, roiid)
                handler.contains(pixid, roiid)
                for shape in roi.iterateShapes():
                    shapeid = handler(shape)
                    handler.annotations(shape, shapeid)
                    handler.contains(roiid, shapeid)
                    if handler.get_sample():
                        break
                if handler.get_sample():
                    break
            return imgid

        else:
            self.ctx.die(111, "unknown target: %s" % target.__class__.__name__)

    def _get_rois(self, gateway, img):
        params = ParametersI()
        params.addId(img.id)
        query = """select r from Roi r
                left outer join fetch r.annotationLinks as ral
                left outer join fetch ral.child as rann
                left outer join fetch r.shapes as s
                left outer join fetch s.annotationLinks as sal
                left outer join fetch sal.child as sann
                     where r.image.id = :id"""
        return gateway.getQueryService().findAllByQuery(
            query, params, {"omero.group": str(img.details.group.id.val)}
        )

    def _lookup(
        self, gateway: BlitzGateway, _type: str, oid: int
    ) -> BlitzObjectWrapper:
        # TODO: move _lookup to a _configure type
        gateway.SERVICE_OPTS.setOmeroGroup("-1")
        obj = gateway.getObject(_type, oid)
        if not obj:
            self.ctx.die(110, f"No such {_type}: {oid}")
        return obj
