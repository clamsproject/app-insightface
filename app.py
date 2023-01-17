"""app.py

Wrapping Spacy NLP to extract tokens, tags, lemmas, sentences, chunks, named
entities, and (if specified so) dependencies.

Usage:

$ python app.py -t input-mmif/example-transcript.json output-mmif/example-transcript.json
$ python app.py [--develop]

The first invocation is to just test the app without running a server. The
second is to start a server, which you can ping with

$ curl -H "Accept: application/json" -X POST -d@input-mmif/example-transcript.json http://0.0.0.0:5000/

With the --develop option you get a FLask server running in development mode,
without it Gunicorn will be used for a more stable server.

Normally you would run this in a Docker container, see README.md.

"""

import os
import sys
import collections
import json
import urllib
import argparse

from infer_on_video import infer_on_video

from clams.app import ClamsApp
from clams.restify import Restifier
from clams.appmetadata import AppMetadata
from mmif.serialize import Mmif
from mmif.vocabulary import AnnotationTypes, DocumentTypes
from lapps.discriminators import Uri

APP_VERSION = '0.0.8'
APP_LICENSE = 'Apache 2.0'
MMIF_VERSION = '0.4.0'
MMIF_PYTHON_VERSION = '0.4.6'
CLAMS_PYTHON_VERSION = '0.5.1'
SPACY_VERSION = '3.3.1'
SPACY_LICENSE = 'MIT'


# We need this to find the text documents in the documents list
VIDEO_DOCUMENT = os.path.basename(str(DocumentTypes.VideoDocument))
Uri_FACE = AnnotationTypes.BoundingBox

DEBUG = False

class InsightFaceApp(ClamsApp):

    def _appmetadata(self):
        
        metadata = AppMetadata(
            identifier='https://apps.clams.ai/insightface',
            url='https://github.com/clamsproject/app-insightface',
            name="Insightface Face Recognition",
            description="Apply InsightFace's ArcFace model to all video documents in a MMIF file.",
            app_version=APP_VERSION,
            app_license=APP_LICENSE,
            analyzer_version=SPACY_VERSION,
            analyzer_license=SPACY_LICENSE,
            mmif_version=MMIF_VERSION
        )
        metadata.add_input(DocumentTypes.VideoDocument)
        metadata.add_output(Uri.ANNOTATION)
        metadata.add_output(Uri_FACE)
        return metadata

    def _annotate(self, mmif, **kwargs):
        Identifiers.reset()
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif, validate=False)
        for doc in video_documents(self.mmif.documents):
            new_view = self._new_view(doc.id)
            self._add_tool_output(doc, new_view)
        for view in list(self.mmif.views):
            docs = self.mmif.get_documents_in_view(view.id)
            docs = video_documents(docs)
            if docs:
                new_view = self._new_view()
                for doc in docs:
                    doc_id = view.id + ':' + doc.id
                    self._add_tool_output(doc, new_view, doc_id=doc_id)
        return self.mmif

    def _new_view(self, docid=None):
        view = self.mmif.new_view()
        self.sign_view(view)
        view.new_contain(Uri.ANNOTATION, document=docid)
        view.new_contain(Uri_FACE, document=docid, timeUnit="frames")
        return view

    def _add_tool_output(self, doc, view, doc_id=None):
        file_name = doc.location # Read the video location from the document
        face_dict = infer_on_video(file_name=file_name)
        #face_dict = infer_on_video(file_name=file_name, save_name="output_video/output.mp4", verbose=True)
        add_annotation(
            view, Uri.ANNOTATION, Identifiers.new("a"), doc_id,
            { "fps": face_dict['fps'], "width": face_dict['width'], "height": face_dict['height'],
              "frame_count": face_dict['frame_count'] })

        bbox_per_frame = face_dict["bounding_boxes_per_frame_index"]
        for i in bbox_per_frame: # 'i' is frame index
            for (name, bbox) in bbox_per_frame[i]:
                [x1, y1, x2, y2] = bbox
                add_annotation(
                    view, Uri_FACE, Identifiers.new("b"), doc_id,
                    { "timepoint": i, "name": name, "coordinates": [[x1,y1], [x2,y1], [x1,y2], [x2,y2]]})

def video_documents(documents):
    """Utility method to get all text documents from a list of documents."""
    return [doc for doc in documents if str(doc.at_type).endswith(VIDEO_DOCUMENT)]

def add_annotation(view, attype, identifier, doc_id, properties):
    """Utility method to add an annotation to a view."""
    a = view.new_annotation(attype, identifier)
    if doc_id is not None:
        a.add_property('document', doc_id)
    for prop, val in properties.items():
        a.add_property(prop, val)

class Identifiers(object):

    """Utility class to generate annotation identifiers. You could, but don't have
    to, reset this each time you start a new view. This works only for new views
    since it does not check for identifiers of annotations already in the list
    of annotations."""

    identifiers = collections.defaultdict(int)

    @classmethod
    def new(cls, prefix):
        cls.identifiers[prefix] += 1
        return "%s%d" % (prefix, cls.identifiers[prefix])

    @classmethod
    def reset(cls):
        cls.identifiers = collections.defaultdict(int)

def test(infile, outfile):
    """Run spacy on an input MMIF file. This bypasses the server and just pings
    the annotate() method on the SpacyApp class. Prints a summary of the views
    in the end result."""
    print(InsightFaceApp().appmetadata(pretty=True))
    with open(infile) as fh_in, open(outfile, 'w') as fh_out:
        mmif = Mmif(fh_in.read(), validate=False)
        mmif_out_as_string = InsightFaceApp().annotate(mmif, pretty=True)
        mmif_out = Mmif(mmif_out_as_string)
        fh_out.write(mmif_out_as_string)
        for view in mmif_out.views:
            print("<View id=%s annotations=%s app=%s>"
                  % (view.id, len(view.annotations), view.metadata['app']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',  action='store_true', help="bypass the server")
    parser.add_argument('--develop',  action='store_true', help="start a development server")
    parser.add_argument('infile', nargs='?', help="input MMIF file")
    parser.add_argument('outfile', nargs='?', help="output file")
    args = parser.parse_args()

    if args.test:
        test(args.infile, args.outfile)
    else:
        app = InsightFaceApp()
        service = Restifier(app)
        if args.develop:
            service.run()
        else:
            service.serve_production()
