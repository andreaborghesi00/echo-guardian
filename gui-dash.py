import numpy as np
from scipy import ndimage
from skimage import data, draw
import cv2 as cv

import dash
from dash import Dash, dcc, html, Input, Output, State, no_update, callback, ctx
from dash_extensions.enrich import DashProxy, LogTransform, DashLogger
from dash.exceptions import PreventUpdate
import dash_daq as daq
import plotly.express as px
from flask import g

import io
import base64
from PIL import Image

from NNClassification import NNClassifier
from UnetSegmenter import UnetSegmenter
from SimpleNet import SimpleNet
import SimpleNet as sn

def path_to_indices(path):
    """
    From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)

def path_to_mask(path, shape):
    """
    From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)

    rr_clipped = np.clip(rr - 1, 0, shape[0] - 1)
    cc_clipped = np.clip(cc - 1, 0, shape[1] - 1)

    mask = np.zeros(shape, dtype=bool)
    mask[rr_clipped, cc_clipped] = True
    mask_numpy = np.array(mask)
    mask = ndimage.binary_fill_holes(mask)
    return mask, mask_numpy

def set_classifier_path(path):
    g.classifier_path = path

def get_classifier_path():
    if 'classifier_path' not in g:
        set_classifier_path('./models/best_model.pth')
    return g.classifier_path

def get_classifier():
    if 'classifier' not in g:
        g.classifier = NNClassifier(get_classifier_path())
    return g.classifier

def set_segmenter_path(path):
    g.segmenter_path = path

def get_segmenter_path():
    if 'segmenter_path' not in g:
        set_segmenter_path('./models/DeepLabV3Plus_resnet34_lr_0.0001_epochs_100_actual_model.pth')
    return g.segmenter_path

def get_segmenter():
    if 'segmenter' not in g:
        g.segmenter = UnetSegmenter(get_segmenter_path())
    return g.segmenter

def predict_roi(img):
    segmenter = get_segmenter()
    return segmenter.predict(img)

def predict_class(img, mask):
    classifier = get_classifier()
    return classifier.predict(img, mask)

def segment_and_classify(image):
    roi_pred = predict_roi(image)
    return predict_class(image, roi_pred)

img = data.chelsea()
mask = np.zeros(img.shape, dtype=bool)
fig = px.imshow(img)
fig_mask = px.imshow(mask)
fig.update_layout(dragmode='drawclosedpath')


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Medical Image Analysis", style={'textAlign': 'center'}),
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
            multiple=True
        ),
        html.Div(id='output-data-upload', style={'marginBottom': '20px'}),
    ], style={'marginBottom': '30px'}),
    html.H3("Draw the Region Of Interest (ROI) on the image below:", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Graph(id="ultrasound-image", figure=fig),
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '0 20px'}),
        html.Div([
            dcc.Graph(id="segmenter-image", figure=fig_mask),
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '0 20px'}),
    ], style={'marginBottom': '30px', 'textAlign': 'center'}),
    html.Div([
        html.Button('Predict with your mask', id='classify-button', n_clicks=0, style={
            'fontSize': '16px',
            'padding': '10px 20px 10px 20px',
            'marginRight': '10px',
            'backgroundColor': '#4CAF50',
            'color': 'white',
            'border': 'none',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'transition': 'background-color 0.3s ease'
        }),
        html.Button('Predict with segmenter mask', id='classify-with-segmenter-button', n_clicks=0, style={
            'fontSize': '16px',
            'padding': '10px 20px 10px 20px',
            'marginLeft': '10px',
            'backgroundColor': '#4CAF50',
            'color': 'white',
            'border': 'none',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'transition': 'background-color 0.3s ease'
        })
        ,
        html.Div(id='output-predict', style={
            'marginTop': '20px',
            'fontFamily': 'monospace',
            'backgroundColor': '#f0f0f0',
            'padding': '10px',
            'borderRadius': '4px'
        }),
    ], style={'textAlign': 'center', 'marginTop': '30px'}),
    dcc.Store(id='image-store'),
    dcc.Store(id='mask-store'),
    dcc.Store(id='segmenter-store'),
], style={
    'padding': '30px',
    'backgroundColor': '#f5f5f5',
    # 'backgroundImage': 'url("background.jpg")',
    'backgroundSize': 'cover',
    'backgroundPosition': 'center'
})

@callback(
    Output("mask-store", "data"),
    Input("ultrasound-image", "relayoutData"),
    State("image-store", "data"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data, image):
    if "shapes" in relayout_data:
        last_shape = relayout_data["shapes"][-1]
        mask_numpy= path_to_mask(last_shape["path"], np.array(image).shape)
        return mask_numpy
    else:
        raise PreventUpdate


@callback(
    Output("ultrasound-image", "figure"),
    Output("image-store", "data"),
    Output("segmenter-image", "figure"),
    Output("segmenter-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def on_upload_data(contents, filenames):
    if contents is not None:
        figures = []
        fig_masks = []
        
        for content, filename in zip(contents, filenames):
            _, content_string = content.split(',')
            decoded_data = base64.b64decode(content_string)

            image_data = io.BytesIO(decoded_data)

            img = Image.open(image_data).convert("L")
            numpy_image = np.array(img)
            numpy_image = cv.resize(numpy_image, (256, 256))
            mask = predict_roi(numpy_image)
            fig_masks.append(px.imshow(mask))

            fig = px.imshow(numpy_image, title=filename)
            fig.update_layout(dragmode='drawclosedpath')
            figures.append(fig)

        # If no figures were created, return an empty figure
        if not figures:
            return px.imshow(np.zeros((1, 1)))

        return figures[0], numpy_image, fig_masks[0], mask
    raise PreventUpdate

@callback(
    Output("output-predict", "children"),
    Input("classify-button", "n_clicks"), # not actually used, it serves as a trigger
    Input("classify-with-segmenter-button", "n_clicks"), # same as above
    State("image-store", "data"),
    State("mask-store", "data"),
    State("segmenter-store", "data"),
    prevent_initial_call=True,
)
def on_predict(n_clicks_classify, n_clicks_classify_segmenter, image, mask, segmenter_mask):
    if ctx.triggered_id == "classify-button":
        if image is None or mask is None:
            return "Please upload an image and draw an ROI before predicting."
        else:
            try:
                class_pred = predict_class(image, mask)
                return f"Predicted class: {class_pred}"
            except Exception as e:
                return f"Error: {str(e)}"
    elif ctx.triggered_id == "classify-with-segmenter-button":
        if image is None:
            return "Please upload an image before predicting."
        else:
            # try:
            class_pred = predict_class(image, segmenter_mask).cpu().numpy()[0][0]
            
            return f'Predicted class with segmenter ROI:\n {f"Malignant {class_pred*100:.2f}%" if np.round(class_pred) == 1 else f"Benign {(1-class_pred)*100:.2f}%"}'
            # except Exception as e:
            #     return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
