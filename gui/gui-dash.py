import numpy as np
from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.express as px
import dash_daq as daq
from skimage import data, draw
from scipy import ndimage

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

img = data.chelsea()
mask = np.zeros(img.shape, dtype=bool)
fig = px.imshow(img)
fig_mask = px.imshow(mask)
fig.update_layout(dragmode='drawclosedpath')


app = Dash(__name__)
app.layout = html.Div(
    [
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

if __name__ == "__main__":
    app.run(debug=True)
