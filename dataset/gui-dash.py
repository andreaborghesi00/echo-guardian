from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.express as px
import dash_daq as daq
from skimage import data


@callback(
    Output("graph-styled-annotations", "figure"),
    Input("opacity-slider", "value"),
    prevent_initial_call=True,
)
def on_style_change(slider_value):
    fig = px.imshow(img)
    fig.update_layout(
        dragmode="drawclosedpath",
        newshape=dict(opacity=slider_value, fillcolor='#808080'),
    )
    return fig



img = data.chelsea()
fig = px.imshow(img)
fig.update_layout(
    dragmode="drawclosedpath",
    newshape=dict(fillcolor="cyan", opacity=0.3, line=dict(color="darkblue", width=8)),
)

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H3("Drag and draw annotations"),
        dcc.Graph(id="graph-styled-annotations", figure=fig),
        html.Pre('Opacity of annotations'),
        dcc.Slider(id="opacity-slider", min=0, max=1, value=0.5, step=0.1, tooltip={'always_visible':True}),
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),

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
    ]
)

# app.layout = html.Div([
#     
# ])


if __name__ == "__main__":
    app.run(debug=True)
