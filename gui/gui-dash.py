import numpy as np
from dash import Dash, dcc, html, Input, Output, State, no_update, callback
import plotly.express as px
import dash_daq as daq
from skimage import data, draw
from scipy import ndimage
from PIL import Image
import io
from dash_extensions.enrich import DashProxy, LogTransform, DashLogger
import base64
from NNClassification import NNClassifier
from UnetSegmenter import UnetSegmenter
from flask import g

def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)

def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=bool)
    mask[rr-1, cc-1] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def set_classifier_path(path):
    g.classifier_path = path

def get_classifier_path():
    if 'classifier_path' not in g:
        set_classifier_path('model.pth')
    return g.classifier_path

def get_classifier():
    if 'classifier' not in g:
        g.classifier = NNClassifier(get_classifier_path())
    return g.classifier

def set_segmenter_path(path):
    g.segmenter_path = path

def get_segmenter_path():
    if 'segmenter_path' not in g:
        set_segmenter_path('model.pth')
    return g.segmenter_path

def get_segmenter():
    if 'segmenter' not in g:
        g.segmenter = UnetSegmenter(get_segmenter_path())
    return g.segmenter

def segment_and_classify(image):
    segmenter = get_segmenter()
    classifier = get_classifier()
    mask = segmenter.predict(image)
    return classifier.predict(image, mask)

def predict_roi_mask(img):
    segmenter = get_segmenter()
    return segmenter.predict(img)

def predict_class(img):
    classifier = get_classifier()
    return classifier.predict(img)

img = data.chelsea()
mask = np.zeros(img.shape, dtype=bool)
fig = px.imshow(img)
fig_mask = px.imshow(mask)
fig.update_layout(dragmode='drawclosedpath')


app = Dash(__name__)
app.layout = html.Div(
    [
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),
        ]),
        html.H3("Draw the Region Of Interest (ROI) on the image below:"),
        html.Div(
            [dcc.Graph(id="ultrasound-image", figure=fig),],
            style={"width": "60%", "display": "inline-block", "padding": "0 0"},
        ),
        html.Div(
            [dcc.Graph(id="mask-image", figure=fig_mask),],
            style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        ),
    ]
)

@callback(
    Output("mask-image", "figure"),
    Input("ultrasound-image", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    if "shapes" in relayout_data:
        last_shape = relayout_data["shapes"][-1]
        mask = path_to_mask(last_shape["path"], img.shape)
        fig_mask = px.imshow(mask)
        return fig_mask
    else:
        return no_update


@callback(
    Output("ultrasound-image", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def on_upload_data(contents, filenames):
    if contents is not None:
        # Create an empty list to store the figures
        figures = []

        for content, filename in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded_data = base64.b64decode(content_string)

            image_data = io.BytesIO(decoded_data)

            img = Image.open(image_data).convert("L")
            numpy_image = np.array(img)

            print(numpy_image.shape)
            print(numpy_image)

            fig = px.imshow(numpy_image, title=filename)
            fig.update_layout(dragmode='drawclosedpath')
            # Append the figure to the list
            figures.append(fig)

        # If no figures were created, return an empty figure
        if not figures:
            return px.imshow(np.zeros((1, 1)))

        # Return the list of figures
        return figures[0]

if __name__ == "__main__":
    app.run(debug=True)
