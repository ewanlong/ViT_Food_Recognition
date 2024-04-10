import gradio as gr
from transformers import pipeline

pipe = pipeline(task="image-classification", 
                model="ewanlong/food_type_image_detection")
                
gr.Interface.from_pipeline(pipe, 
                           title="Food Recognition",
                           description="Food Categories Recognition",
                           examples=['apple_pie.jpg', 'dhokla.jpg'],
                           article = "Author: <a href=\"https://huggingface.co/ewanlong\">Ewan Long</a>",
                           ).launch()