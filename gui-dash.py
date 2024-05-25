import numpy as np
from scipy import ndimage
from skimage import data, draw
import cv2 as cv

import dash
import dash_bootstrap_components as dbc
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

img = data.immunohistochemistry()
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

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

login_layout = dbc.Container([
    html.Div([
        html.H1("Login", className="display-3"),
        html.Hr(className="my-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Username", html_for="username-input"),
                dbc.Input(
                    id="username-input",
                    type="text",
                    placeholder="Enter your username",
                    className="mb-3",
                    value=""
                ),
                dbc.Label("Password", html_for="password-input"),
                dbc.Input(
                    id="password-input",
                    type="password",
                    placeholder="Enter your password",
                    className="mb-3",
                    value=""
                ),
                dbc.Button("Login", id="login-button", color="primary", className="mb-3"),
                html.Div(id="login-output"),
            ], width=6, className="mx-auto")
        ]),
    ]),
    dcc.Store(id='auth-store', storage_type='session'),
], fluid=True, className="py-3")

main_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Medical Image Analysis", className="display-3"),
        ], width=10, className="mr-auto"),
        dbc.Col([
            dbc.Button("Logout", id="logout-button", color="danger", className="text-center", style={"margin-top": "20px", "margin-left": "30%"}),
        ], width=2, className="ml-auto"),
    ], className="mb-3"),
    html.Hr(className="my-2"),
    dbc.Row([
        dbc.Col([
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
                multiple=False,
                accept='image/png'
            ),
            html.Div(id='output-data-upload', className="mt-3"),
        ], width=12),
    ], className="mt-3"),
    html.Hr(className="my-2"),
    dbc.Row([
        dbc.Col([
            html.H3("Draw the Region Of Interest (ROI) on the image below:"),
            dcc.Graph(id="ultrasound-image", figure=fig, config=config_image),
        ], width=6),
        dbc.Col([
            html.H3("Automatic segmenter mask:"),
            dcc.Graph(id="segmenter-image", figure=fig_mask, config=config_mask),
        ], width=6),
    ], className="mt-3"),
    html.Hr(className="my-2"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict with your mask", id="classify-button", color="primary", className="mr-2", style={"margin-right": "10px"}),
            dbc.Button("Predict with segmenter mask", id="classify-with-segmenter-button", color="secondary", className="ml-2", style={"margin-left": "10px"}),
            html.Div(id="output-predict", className="mt-3"),
        ], width=12, className="text-center"),
    ], className="mt-3"),
    dcc.Store(id='image-store'),
    dcc.Store(id='mask-store'),
    dcc.Store(id='segmenter-store'),
    dcc.Store(id='auth-store'),
    dcc.ConfirmDialog(
        id='confirm-auto-segmenter',
        message='Danger danger! Are you sure you want to continue?',
    ),
    html.Div(id='dummy-input', children=''),
    
], fluid=True, className="py-3")

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='auth-store', storage_type='session'),
])

@app.callback(Output('page-content', 'children', allow_duplicate=True),
              [Input('url', 'pathname')],
              [State('auth-store', 'data')],
              prevent_initial_call='initial_duplicate')
def display_page(pathname, auth):
    if pathname == '/':
        if auth is None or auth['username'] is None or auth['password'] is None:
            return login_layout
        else:
            return main_layout
    elif pathname == '/login':
        return login_layout
    elif pathname == '/main':
        return main_layout
    else:
        return '404 Page Not Found'


@app.callback(
    Output("mask-store", "data"),
    Output("output-predict", "children", allow_duplicate=True),
    Input("ultrasound-image", "relayoutData"),
    State("image-store", "data"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data, image):
    if "shapes" in relayout_data:
        if image is None: return no_update, "Please upload an image before predicting."
        try:
            last_shape = relayout_data["shapes"][-1]
        except IndexError:
            return no_update, "Please upload an image and draw an ROI before predicting."
        
        mask, mask_numpy= path_to_mask(last_shape["path"], np.array(image).shape)
        # squeeze the first dimension
        return mask_numpy, ""
    else:
        raise PreventUpdate


@app.callback(
    Output("ultrasound-image", "figure"),
    Output("image-store", "data"),
    Output("segmenter-image", "figure"),
    Output("segmenter-store", "data"),
    Output("output-predict", "children", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("auth-store", "data"),
    prevent_initial_call=True,
)
def on_upload_data(contents, filename, auth):
    if auth is None or auth['username'] is None or auth['password'] is None: raise PreventUpdate

    if contents is not None:
        if not filename.endswith('png'): return no_update, no_update, no_update, no_update, "File type unsupported, please upload a valid image file."
        
        _, content_string = contents.split(',')
        decoded_data = base64.b64decode(content_string)

        # try: 
        img = Image.open(io.BytesIO(decoded_data)).convert("L")
        # except Exception as e:
        #     return no_update, no_update, no_update, no_update, f"File type unsupported, please upload a valid image file."
        
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

        fig_mask = (px.imshow(mask_numpy, title="Automatic segmenter mask", color_continuous_scale='gray'))
        fig_image = px.imshow(img_numpy, title=filename, color_continuous_scale='gray')
        fig_image.update_layout(dragmode='drawclosedpath')
        # figures.append(fig_image)

        # If no figures were created, return an empty figure
        if not fig_image:
            raise PreventUpdate
        return fig_image, img_numpy, fig_mask, mask_numpy, ""
    raise PreventUpdate


@app.callback(
    Output("output-predict", "children", allow_duplicate=True),
    Output("confirm-auto-segmenter", "displayed"),
    Input("classify-button", "n_clicks"), # not actually used, it serves as a trigger
    Input("classify-with-segmenter-button", "n_clicks"), # same as above
    Input("confirm-auto-segmenter", "submit_n_clicks"),
    State("image-store", "data"),
    State("mask-store", "data"),
    State("segmenter-store", "data"),
    State("auth-store", "data"),
    prevent_initial_call=True,
)
def on_predict(n_clicks_classify, n_clicks_classify_segmenter, confirm_danger_clicks,image, mask, segmenter_mask, auth):
    if auth is None: return "Please login first.", False

    if ctx.triggered_id == "classify-button":
        if image is None or mask is None:
            return "Please upload an image and draw an ROI before predicting.", False
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

                return f'Predicted class with user segmentation:\n {f"Malignant {class_pred*100:.2f}%" if np.round(class_pred) == 1 else f"Benign {(1-class_pred)*100:.2f}%"}', False
            except requests.exceptions.RequestException as e:
                return f"Error: {str(e)}", False
            
    elif ctx.triggered_id == "classify-with-segmenter-button":
        if image is None:
            return "Please upload an image before predicting.", False
        else:            
            return f"", True
    elif ctx.triggered_id == "confirm-auto-segmenter":
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
            return f'Predicted class with auto segmentation:\n {f"Malignant {class_pred*100:.2f}%" if np.round(class_pred) == 1 else f"Benign {(1-class_pred)*100:.2f}%"}', False
        except Exception as e:
            return f"Error: {str(e)}", False


@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Output('auth-store', 'data'),
    Output('login-output', 'children'),
    Input('login-button', 'n_clicks'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    prevent_initial_call=True,
)
def login(n_clicks, username, password):
    if n_clicks is None:
        raise PreventUpdate
    else:
        if not username or not password:
            return login_layout, None, "Unauthorized access"
        
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        auth = HTTPBasicAuth(username, hashed_pw)
        response = requests.post('http://localhost:5000/api/login', auth=auth)
        if response.status_code == 200:
            print('Successfully logged in')
            return main_layout, {'username': username, 'password': hashed_pw}, ""
        else:
            print('Unauthorized access')
            return login_layout, None, "Unauthorized access"


@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Input('logout-button', 'n_clicks'),
    State('auth-store', 'data'),
    prevent_initial_call=True,
)
def logout(n_clicks, auth):
    if n_clicks is not None and n_clicks > 0:
        # auth['username'], auth['password'] = None, None
        return login_layout
    else:
        raise PreventUpdate
    
@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Input('dummy-input', 'children'),
    State('auth-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def check_if_logged_in(dummy, auth):
    if auth is None or auth['username'] is None or auth['password'] is None:
        return login_layout
    else:
        return main_layout
        
if __name__ == '__main__':
    app.run_server(debug=True)