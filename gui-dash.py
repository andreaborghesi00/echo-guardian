import numpy as np
from scipy import ndimage
from skimage import data, draw
import cv2 as cv

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, no_update, callback, ctx
from dash_extensions.enrich import DashProxy, LogTransform, DashLogger
import plotly.graph_objects as go
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

import cv2
from svgpathtools import Path, Line

class SvgPath:
    def __init__(self, contour, color='#444'):
        self.contour = contour
        self.color = color
        self.path = self.generate_path()

    def generate_path(self):
        if len(self.contour) == 0:
            return ""
        path = "M"
        for i in range(len(self.contour)):
            x, y = self.contour[i][0]
            if i == len(self.contour) - 1:
                path += f"{float(x)},{float(y)}"
            else:
                path += f"{float(x)},{float(y)}L"
        path += "Z"
        return path

    def to_dict(self):
        return {
            "editable": True,
            "label": {"text": ""},
            "xref": "x",
            "yref": "y",
            "layer": "above",
            "opacity": 1,
            "line": {"color": self.color, "width": 4, "dash": "solid"},
            "fillcolor": "rgba(0,0,0,0)",
            "fillrule": "evenodd",
            "type": "path",
            "path": self.path
        }

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css',
    {
        'href': 'https://cdn.jsdelivr.net/gh/creativetimofficial/tailwind-starter-kit/compiled-tailwind.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha256-Qw8g/pxKUoW+1eWoEAMBYRl6BGh9/ynIo/XMpgi+Zw=',
        'crossorigin': 'anonymous'
    },
    {
        'href': 'https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap',
        'rel': 'stylesheet'
    }
]

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
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
    ],
    "doubleClick": "reset",
    "showTips": False,
    "showAxisDragHandles": False,
    "showAxisRangeEntryBoxes": False,
    "scrollZoom": False,
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

external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Login", className="display-3 text-center")
        ], width=12)
    ]),
    html.Hr(className="my-2"),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Username", html_for="username-input", className="text-center")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Input(
                        id="username-input",
                        type="text",
                        placeholder="Enter your username",
                        className="form-control mb-3",
                        value=""
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Password", html_for="password-input", className="text-center")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Input(
                        id="password-input",
                        type="password",
                        placeholder="Enter your password",
                        className="form-control mb-3",
                        value="",
                        debounce=True,
                        n_submit=0
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Login", id="login-button", color="primary", className="mb-3")
                ], width=12, className="text-center")
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="login-output", className="text-center")
                ], width=12)
            ]),
        ], width=6, className="mx-auto")
    ]),
    dcc.Store(id='auth-store', storage_type='session'),
], fluid=True, className="py-3")


with open("images/dragndrop.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
icon_base64 = f"data:image/png;base64,{encoded_string}"

main_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Medical Image Analysis", className="display-4 text-center mb-4", style={"font-weight": "bold"})
        ], width=12)
    ], justify="center", className="text-center"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Logout", id="logout-button", color="danger", className="text-center", style={"margin-top": "10px", "margin-right": "10px"})
        ], style={"display": "flex", "flex-flow": "row wrap", "justify-content": "flex-end"})
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "end"}),
    html.Hr(className="my-2"),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Div([
                        html.H4("Drag and Drop or Select Image", className="mb-2"),
                        html.Img(src=icon_base64, style={'width': '48px', 'height': '48px'}),
                    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'height': '100%'}),
                ]),
                style={
                    'width': '100%',
                    'height': '150px',
                    'lineHeight': '150px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'cursor': 'pointer',
                    'background-color': '#fff',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
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
            dbc.Row([
                dbc.Col([
                    html.H3("Draw the Region Of Interest (ROI) on the image below:", className="text-center")
                ], width=12)
            ]),
            dcc.Graph(id="ultrasound-image", figure=fig, config=config_image, className="graph-figure"),
        ], width=12, md=6, className="text-center"),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H3("Automatic segmenter mask:", className="text-center")
                ], width=12)
            ]),
            dcc.Graph(id="segmenter-image", figure=fig_mask, config=config_mask, className="graph-figure"),
        ], width=12, md=6, className="text-center"),
    ], className="mt-3", justify="center"),
    html.Hr(className="my-2"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict with your mask", id="classify-button", color="primary", className="mr-2", style={"margin-right": "10px"}),
            dbc.Button("Predict with segmenter mask", id="classify-with-segmenter-button", color="secondary", className="ml-2", style={"margin-left": "10px"}),
            dbc.Button("Load Segmenter Mask", id="load-segmenter-mask-button", color="info", className="ml-2", style={"margin-left": "10px"}),
            html.Div(id="output-predict", className="mt-3"),
        ], width=12, className="text-center"),
    ], className="mt-3"),
    dbc.Row([
        dbc.Col([
            dbc.Spinner(color="primary", type="grow"),
            html.Div("Loading segmented mask...", id="loading-output")
        ], width=12, className="text-center")
    ], className="mt-3", justify="center", id="loading-row", style={"display": "none"}),
    dcc.Store(id='image-store'),
    dcc.Store(id='mask-store'),
    dcc.Store(id='segmenter-store'),
    dcc.Store(id='auth-store'),
    dcc.ConfirmDialog(
        id='confirm-auto-segmenter',
        message='Are you sure you want to continue? The autosegmenter mask will be used for prediction, it may not be accurate.',
    ),
    html.Div(id='dummy-input', children=''),
], fluid=True, className="py-3", style={"background-color": "#f8f9fa"})


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='auth-store', storage_type='session'),
], style={'background-color': '#f8f9fa', 'min-height': '100vh'})

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
    Output("ultrasound-image", "figure", allow_duplicate=True),
    Output("image-store", "data"),
    Output("segmenter-image", "figure"),
    Output("segmenter-store", "data"),
    Output("loading-row", "style"),
    Input("ultrasound-image", "relayoutData"),
    Input("load-segmenter-mask-button", "n_clicks"),
    Input("upload-data", "contents"),
    State("image-store", "data"),
    State("mask-store", "data"),
    State("segmenter-store", "data"),
    State("ultrasound-image", "figure"),
    State("upload-data", "filename"),
    State("auth-store", "data"),
    prevent_initial_call=True,
)
def update_annotation_and_upload(relayout_data, load_segmenter_mask_clicks, contents, image, current_mask, segmenter_mask, current_figure, filename, auth):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "ultrasound-image":
        if "shapes" in relayout_data:
            if image is None:
                return no_update, "Please upload an image before predicting.", no_update, no_update, no_update, no_update, {"display": "none"}
            
            shapes = relayout_data["shapes"]
            if len(shapes) == 0:
                # No shapes, clear the mask
                return np.zeros(np.array(image).shape, dtype=bool), "", current_figure, no_update, no_update, no_update, {"display": "none"}
            else:
                # Use the last shape to create the mask
                last_shape = shapes[-1]
                mask, mask_numpy = path_to_mask(last_shape["path"], np.array(image).shape)
                return mask_numpy, "", current_figure, no_update, no_update, no_update, {"display": "none"}
        elif "shapes[0].path" in relayout_data:
            if image is None:
                return no_update, "Please upload an image before predicting.", no_update, no_update, no_update, no_update, {"display": "none"}
            
            if current_mask is None:
                # If there is no current mask, create a new one
                path = relayout_data["shapes[0].path"]
                mask, mask_numpy = path_to_mask(path, np.array(image).shape)
                return mask_numpy, "", current_figure, no_update, no_update, no_update, {"display": "none"}
            else:
                # If there is a current mask, update it with the new path
                path = relayout_data["shapes[0].path"]
                mask, mask_numpy = path_to_mask(path, np.array(image).shape)
                return mask_numpy, "", current_figure, no_update, no_update, no_update, {"display": "none"}
        else:
            raise PreventUpdate

    elif trigger_id == "load-segmenter-mask-button":
        if segmenter_mask is None or image is None:
            return no_update, no_update, current_figure, no_update, no_update, no_update, {"display": "none"}

        # Convert segmenter_mask to a NumPy array if it is a list
        if isinstance(segmenter_mask, list):
            segmenter_mask = np.array(segmenter_mask)

        # Transform the segmenter mask into a shape
        contours, _ = cv2.findContours(segmenter_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        svg_paths = []
        for c in contours:
            svg_path = SvgPath(c, color='blue')  # Pass the color 'blue' here
            svg_paths.append(svg_path.to_dict())

        # Create a new figure with the updated shape
        new_figure = go.Figure(data=[go.Heatmap(z=np.array(image), colorscale=[[0, 'black'], [1, 'white']])])
        new_figure.update_layout(
            shapes=svg_paths,
            dragmode='drawclosedpath',
            newshape=dict(line_color='blue', fillcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, b=0, t=0),
            xaxis=dict(visible=False, range=[0, segmenter_mask.shape[1]]),
            yaxis=dict(visible=False, range=[segmenter_mask.shape[0], 0], scaleanchor="x", scaleratio=1)
        )

        # Update the line color of the loaded segmenter mask
        for path in svg_paths:
            path['line']['color'] = 'blue'

        # Create figure for segmenter mask
        fig_mask = px.imshow(segmenter_mask, color_continuous_scale=[[0, 'black'], [1, 'white']], aspect="equal")
        fig_mask.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            autosize=True,
            height=500
        )
        fig_mask.update_xaxes(showticklabels=False)
        fig_mask.update_yaxes(showticklabels=False)

        return segmenter_mask, "", new_figure, no_update, fig_mask, no_update, {"display": "none"}

    elif trigger_id == "upload-data":
        if auth is None or auth['username'] is None or auth['password'] is None:
            raise PreventUpdate

        if contents is not None:
            if not filename.endswith('png'):
                return no_update, "File type unsupported, please upload a valid image file.", no_update, no_update, no_update, no_update, {"display": "none"}
            
            _, content_string = contents.split(',')
            decoded_data = base64.b64decode(content_string)

            img = Image.open(io.BytesIO(decoded_data)).convert("L")
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            img_numpy = np.array(img)
            img_numpy = cv.resize(img_numpy, (256, 256))

            files = {'image': ("Image.png", img_bytes, "image/png")}

            # Create figure for ultrasound image
            fig_image = go.Figure(data=[go.Heatmap(z=img_numpy, colorscale=[[0, 'black'], [1, 'white']])])
            fig_image.update_layout(
                dragmode='drawclosedpath',
                newshape=dict(line_color='blue', fillcolor='rgba(0,0,0,0)'),
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                height=500,
                xaxis=dict(visible=False, range=[0, img_numpy.shape[1]]),
                yaxis=dict(visible=False, range=[img_numpy.shape[0], 0], scaleanchor="x", scaleratio=1)
            )

            # Show loading screen while segmentation is being processed
            loading_figure = go.Figure(
                data=[go.Scatter()],
                layout=go.Layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=0, r=0, b=0, t=0),
                    autosize=True,
                    height=500,
                    template='plotly_white'
                )
            )
            loading_figure.update_layout(
                shapes=[
                    dict(
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=-1,
                        y0=-1,
                        x1=1,
                        y1=1,
                        line_color="lightgrey",
                        line_width=2
                    )
                ],
                annotations=[
                    dict(
                        text="Loading segmentation...",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font_size=20,
                        font_color="lightgrey"
                    )
                ]
            )

            response = requests.post('http://localhost:5000/api/segment', files=files, auth=HTTPBasicAuth(username=auth['username'], password=auth['password']))
            response.raise_for_status()

            mask_bytes = response.content
            mask = Image.open(io.BytesIO(mask_bytes))
            mask_numpy = np.array(mask)
            mask_numpy = ndimage.binary_fill_holes(mask_numpy)

            # Create figure for segmenter mask
            fig_mask = px.imshow(mask_numpy, color_continuous_scale=[[0, 'black'], [1, 'white']], aspect="equal")
            fig_mask.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True,
                height=500
            )
            fig_mask.update_xaxes(showticklabels=False)
            fig_mask.update_yaxes(showticklabels=False)

            if not fig_image:
                raise PreventUpdate
            return no_update, "", fig_image, img_numpy, fig_mask, mask_numpy, {"display": "none"}
        raise PreventUpdate

    else:
        raise PreventUpdate



@app.callback(
    Output("output-predict", "children", allow_duplicate=True),
    Output("confirm-auto-segmenter", "displayed"),
    Input("classify-button", "n_clicks"),
    Input("classify-with-segmenter-button", "n_clicks"),
    Input("confirm-auto-segmenter", "submit_n_clicks"),
    State("image-store", "data"),
    State("mask-store", "data"),
    State("segmenter-store", "data"),
    State("auth-store", "data"),
    prevent_initial_call='initial_duplicate'
)
def on_predict(n_clicks_classify, n_clicks_classify_segmenter, confirm_danger_clicks, image, mask, segmenter_mask, auth):
    if auth is None:
        return "Please login first.", False
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "classify-button" or (n_clicks_classify is not None and n_clicks_classify > 0):
        if image is None:
            return "Please upload an image before predicting.", False
        elif mask is None or np.all(mask == 0):
            return "Please draw an ROI before predicting.", False
        else:
            try:
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                if not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

                if np.all(mask == 0):
                    return "Please draw a ROI before predicting.", False

                image = Image.fromarray(image.astype('uint8')).convert("L")
                mask = Image.fromarray(mask.astype('uint8')).convert("L")

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

    elif trigger_id == "classify-with-segmenter-button":
        if image is None:
            return "Please upload an image before predicting.", False
        else:
            return "", True

    elif trigger_id == "confirm-auto-segmenter":
        try:
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            if not isinstance(segmenter_mask, np.ndarray):
                segmenter_mask = np.array(segmenter_mask)

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
    if auth is None:
        return "Please login first.", False

    if trigger_id == "classify-button":
        if image is None:
            return "Please upload an image before predicting.", False
        elif mask is None or np.all(mask == 0):
            return "Please draw an ROI before predicting.", False
        else:
            try:
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                if not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

                image = Image.fromarray(image.astype('uint8')).convert("L")
                mask = Image.fromarray(mask.astype('uint8')).convert("L")

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

    elif trigger_id == "classify-with-segmenter-button":
        if image is None:
            return "Please upload an image before predicting.", False
        else:
            return "", True

    elif trigger_id == "confirm-auto-segmenter":
        try:
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            if not isinstance(segmenter_mask, np.ndarray):
                segmenter_mask = np.array(segmenter_mask)

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
    if auth is None: return "Please login first.", False

    if trigger_id == "classify-button":
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
                if response.status_code == 500:
                    return "Error: Please draw a valid ROI before predicting.", False
                response.raise_for_status()

                class_pred = response.json()['prediction']

                return f'Predicted class with user segmentation:\n {f"Malignant {class_pred*100:.2f}%" if np.round(class_pred) == 1 else f"Benign {(1-class_pred)*100:.2f}%"}', False
            except requests.exceptions.RequestException as e:
                return f"Error: {str(e)}", False
            
    elif trigger_id == "classify-with-segmenter-button":
        if image is None:
            return "Please upload an image before predicting.", False
        else:            
            return f"", True
    elif trigger_id == "confirm-auto-segmenter":
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