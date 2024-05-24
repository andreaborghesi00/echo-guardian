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
import requests
from requests.auth import HTTPBasicAuth


import io
import base64
import hashlib
from PIL import Image

from NNClassification import NNClassifier
from UnetSegmenter import UnetSegmenter

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
        set_classifier_path('models/best_model.pth')
    return g.classifier_path

def get_classifier():
    if 'classifier' not in g:
        g.classifier = NNClassifier(get_classifier_path())
    return g.classifier

def set_segmenter_path(path):
    g.segmenter_path = path

def get_segmenter_path():
    if 'segmenter_path' not in g:
        set_segmenter_path('models/DeepLabV3Plus_resnet34_lr_0.0001_epochs_100_actual_model.pth')
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

config_image = {
    "modeBarButtonsToAdd": [
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ], 
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "pan2d",
        "lasso2d",
        "select2d",
    ],
}

config_mask = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "pan2d",
        "lasso2d",
        "select2d",
    ],
}

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Medical Image Analysis", style={'textAlign': 'center'}),
    html.Div(id='login-section'),  # A placeholder div for the login section
    dcc.Input(id='dummy-input', value='initial-value', style={'display': 'none'}),
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
            dcc.Graph(id="ultrasound-image", figure=fig, config=config_image),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 2px'}),
        html.Div([
            dcc.Graph(id="segmenter-image", figure=fig_mask, config=config_mask),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 2px'}),
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
    dcc.Store(id='auth-store'),
], style={
    'padding': '10px',
    'backgroundColor': '#f5f5f5',
    # 'backgroundImage': 'url("background.jpg")',
    'backgroundSize': 'cover',
    'backgroundPosition': 'center'
})

login_section = html.Div([
            dcc.Input(
                id='username-input',
                type='text',
                placeholder='Enter your username',
                style={'width': '200px', 'marginRight': '10px'}
            ),
            dcc.Input(
                id='password-input',
                type='password',
                placeholder='Enter your password',
                style={'width': '200px', 'marginRight': '10px'}
            ),
            html.Button('Login', id='login-button', n_clicks=0, style={
                'fontSize': '16px',
                'padding': '5px 10px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'transition': 'background-color 0.3s ease'
            }),
            html.Div(id='login-output', style={'marginTop': '10px'})
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),

logged_in_section = html.Div([
            html.Button('Logout', id='logout-button', n_clicks=0, style={
            'fontSize': '16px',
            'padding': '5px 10px',
            'backgroundColor': '#AF4C4C',
            'color': 'white',
            'border': 'none',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'transition': 'background-color 0.3s ease'
        })
    ], style={'marginBottom': '20px', 'textAlign': 'center'})

@callback(
    Output("mask-store", "data"),
    Input("ultrasound-image", "relayoutData"),
    State("image-store", "data"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data, image):
    if "shapes" in relayout_data:
        last_shape = relayout_data["shapes"][-1]
        mask, mask_numpy= path_to_mask(last_shape["path"], np.array(image).shape)
        # squeeze the first dimension
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
    State("auth-store", "data"),
    prevent_initial_call=True,
)
def on_upload_data(contents, filenames, auth):
    if auth is None: no_update, no_update, no_update, no_update, "Please login first."

    if contents is not None:
        figures = []
        fig_masks = []
        
        for content, filename in zip(contents, filenames):
            _, content_string = content.split(',')
            decoded_data = base64.b64decode(content_string)

            img = Image.open(io.BytesIO(decoded_data)).convert("L")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            img_numpy = np.array(img)
            img_numpy = cv.resize(img_numpy, (256, 256))

            files = {'image': ("Image.png", img_bytes, "image/png")}

            response = requests.post('http://localhost:5000/api/segment', files=files, auth=HTTPBasicAuth(username=auth['username'], password=auth['password']))
            response.raise_for_status()

            mask_bytes = response.content
            mask = Image.open(io.BytesIO(mask_bytes))
            mask_numpy = np.array(mask)
            mask_numpy = ndimage.binary_fill_holes(mask_numpy)

            print(mask_numpy.dtype)

            fig_masks.append(px.imshow(mask_numpy, title="Automatic segmenter mask", color_continuous_scale='gray'))
            fig_image = px.imshow(img_numpy, title=filename, color_continuous_scale='gray')
            fig_image.update_layout(dragmode='drawclosedpath')
            figures.append(fig_image)

        # If no figures were created, return an empty figure
        if not figures:
            raise PreventUpdate
        return figures[0], img_numpy, fig_masks[0], mask_numpy
    raise PreventUpdate


@callback(
    Output("output-predict", "children"),
    Input("classify-button", "n_clicks"), # not actually used, it serves as a trigger
    Input("classify-with-segmenter-button", "n_clicks"), # same as above
    State("image-store", "data"),
    State("mask-store", "data"),
    State("segmenter-store", "data"),
    State("auth-store", "data"),
    prevent_initial_call=True,
)
def on_predict(n_clicks_classify, n_clicks_classify_segmenter, image, mask, segmenter_mask, auth):
    if auth is None: return "Please login first."

    if ctx.triggered_id == "classify-button":
        if image is None or mask is None:
            return "Please upload an image and draw an ROI before predicting."
        else:
            try:
                if not isinstance(image, np.ndarray): image = np.array(image)
                if not isinstance(mask, np.ndarray): mask = np.array(mask)

                image = Image.fromarray(image.astype('uint8')).convert("L")
                mask = Image.fromarray(mask.astype('uint8')).convert("L")

                # Convert the image and mask to bytes
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes.seek(0)

                mask_bytes = io.BytesIO()
                mask.save(mask_bytes, format='PNG')
                mask_bytes.seek(0)

                files = {'image': image_bytes, 'mask': mask_bytes}
                response = requests.post('http://localhost:5000/api/classify', files=files, auth=HTTPBasicAuth(username=auth['username'], password=auth['password']))
                response.raise_for_status()

                class_pred = response.json()['prediction']

                return f'Predicted class with user segmentation:\n {f"Malignant {class_pred*100:.2f}%" if np.round(class_pred) == 1 else f"Benign {(1-class_pred)*100:.2f}%"}'
            except requests.exceptions.RequestException as e:
                return f"Error: {str(e)}"
            
    elif ctx.triggered_id == "classify-with-segmenter-button":
        if image is None:
            return "Please upload an image before predicting."
        else:
            try:
                if not isinstance(image, np.ndarray): image = np.array(image) # for some reason it's a python list although i save it as numpy array
                if not isinstance(segmenter_mask, np.ndarray): segmenter_mask = np.array(segmenter_mask) # same here
                
                image = Image.fromarray(image.astype('uint8')).convert("L") 
                mask = Image.fromarray(segmenter_mask.astype('uint8')).convert("L") 

                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes.seek(0)

                mask_bytes = io.BytesIO()
                mask.save(mask_bytes, format='PNG')
                mask_bytes.seek(0)

                files = {'image': image_bytes, 'mask': mask_bytes}
                response = requests.post('http://localhost:5000/api/classify', files=files, auth=HTTPBasicAuth(username=auth['username'], password=auth['password']))
                response.raise_for_status()

                class_pred = response.json()['prediction']

                return f'Predicted class with segmenter ROI:\n {f"Malignant {class_pred*100:.2f}%" if np.round(class_pred) == 1 else f"Benign {(1-class_pred)*100:.2f}%"}'
            except Exception as e:
                return f"Error: {str(e)}"

# don't judge me for this first render
@app.callback(
        Output('login-section', 'children', allow_duplicate=True),
        Input('dummy-input', 'value'),  # A dummy input to trigger the callback
        suppress_callback_exceptions=True,
        prevent_initial_call='initial duplicate',
)
def render_first_login_section(dummy):
    global login_section
    return login_section

@app.callback(
    Output('login-section', 'children', allow_duplicate=True),
    Output('auth-store', 'data'),
    Input('login-button', 'n_clicks'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    prevent_initial_call=True,
)
def render_login_section(n_clicks, username, password):
    global login_section, logged_in_section
    if n_clicks is None: raise PreventUpdate

    if not username or not password:
        return login_section, None

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    auth = HTTPBasicAuth(username, hashed_pw)
    response = requests.post('http://localhost:5000/api/login', auth=auth)
    if response.status_code == 200:
        print('Successfully logged in')
        return logged_in_section, {'username': username, 'password': hashed_pw}
    else:
        print('Unauthorized access')
        return login_section, None

@app.callback(
    Output('login-section', 'children'),
    Input('logout-button', 'n_clicks'),
    State('auth-store', 'data'),
    prevent_initial_call=True,
)
def on_logout(n_clicks, auth):
    if n_clicks > 0:
        auth['username'], auth['password'] = None, None
        return login_section
    else: raise PreventUpdate




if __name__ == "__main__":
    app.run(debug=True)
