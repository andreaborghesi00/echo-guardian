# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %% [GUI]
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import numpy as np
import cv2 as cv
from common import Sketcher
from streamlit_js_eval import streamlit_js_eval
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")

# @see https://github.com/streamlit/streamlit/issues/708#issuecomment-1678883348
# Save viewport width to session state
st.session_state["screen_width"] = streamlit_js_eval(
    js_expressions="screen.width", key="ScreenWidth"
)
st.session_state["screen_height"] = streamlit_js_eval(
    js_expressions="screen.height", key="ScreenHeight"
)
print(st.session_state["screen_width"], st.session_state['screen_height'])

# Define layout
col1, col2 = st.columns([0.2, 0.8])

with col1: 
    st.title('Tumor detector')

    # Load image
    uploaded_file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg', 'tiff', 'tif'])

with col2:
    # if uploaded_file is not None:
    #     # img = cv.imread(uploaded_file)
    #     img = Image.open(uploaded_file)
    #     img = np.array(img)
    #     img_mark = img.copy()
    #     mark = np.zeros(img.shape[:2], np.uint8)
    #     sketch = Sketcher('img', [img_mark, mark], lambda : ((255, 255, 255), 255))

    #     while True:
    #         ch = cv.waitKey()
    #         if ch == 27:
    #             break
    #         if ch == ord(' '):
    #             res = cv.inpaint(img_mark, mark, 3, cv.INPAINT_TELEA)
    #             cv.imshow('inpaint', res)
    #             st.image(res, caption='Inpainted Image', use_column_width=True)  # Display the image using Streamlit
    #         if ch == ord('r'):
    #             img_mark[:] = img
    #             mark[:] = 0
    #             sketch.show()

    # Calculate width and height based on viewport width
    width = st.session_state["screen_width"] * 0.8
    height = st.session_state["screen_height"]

    fig = go.Figure()
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        if img.size[0] > img.size[1]:
            scaling_factor = (width * 0.8) / img.size[0]
        else:
            scaling_factor = (height * 0.8) / img.size[1]

        new_size = (int(img.size[0] * scaling_factor), int(img.size[1] * scaling_factor))

        print(new_size)
        img = img.resize(new_size)

        fig = px.imshow(img)
        fig.update_layout(
            dragmode="drawclosedpath",
            newshape_line_color="cyan",
            title_text="Select the region of interest",
            height=height,
        )

    config = dict(
        {
            "scrollZoom": True,
            "displayModeBar": False,
            # 'editable'              : True,
            "modeBarButtonsToAdd": [
                "drawline", 
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
            "toImageButtonOptions": {"format": "svg"},
        })

    
    st.plotly_chart(fig, config=config, use_container_width=True)
