from flask import Flask, request, jsonify, Response, send_file
import base64
from requests.utils import requote_uri
"""app.main: handle request for lambda-tiler"""

import re
import json

import numpy as np
from flask_compress import Compress
from flask_cors import CORS

from rio_tiler import main
from rio_tiler.utils import (array_to_image,
                             linear_rescale,
                             get_colormap,
                             expression,
                             mapzen_elevation_rgb)

from src import cmap
from src import elevation as ele_func
from src import value as get_value
from src import response
from src import profile as get_profile
from src import volume as get_volume

import time
import gzip
# from lambda_proxy.proxy import API


class TilerError(Exception):
    """Base exception class."""


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

app.config['COMPRESS_MIMETYPES'] = ['text/html', 'text/css', 'text/xml',
                                    'application/json',
                                    'application/javascript',
                                    'image/png',
                                    'image/PNG',
                                    'image/jpg',
                                    'imgae/jpeg',
                                    'image/JPG',
                                    'image/JPEG']
app.config['COMPRESS_LEVEL'] = 9
app.config['COMPRESS_MIN_SIZE'] = 0
Compress(app)


# Welcome page
@app.route('/')
def hello():
    return "Welcome to, COG API!"


# Generates bounds of Raster data
@app.route('/api/v1/bounds', methods=['GET'])
def bounds():
    """Handle bounds requests."""
    url = request.args.get('url', default='', type=str)
    url = requote_uri(url)

    # address = query_args['url']
    info = main.bounds(url)
    return (jsonify(info))


# Generates metadata of raster
@app.route('/api/v1/metadata', methods=['GET'])
def metadata():
    """Handle metadata requests."""
    url = request.args.get('url', default='', type=str)
    url = requote_uri(url)

    # address = query_args['url']
    info = main.metadata(url)
    return (jsonify(info))