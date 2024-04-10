from transformers import pipeline
import os
import gradio as gr
import openai

# Use environment variables for security best practices
openai.api_key = 'ENTER_YOUR_OPENAI_API_KEY_HERE'

def generate_text_with_gpt(prompt):
    try:
        # Adjusted for the chat model endpoint
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # Specify the appropriate chat model
            messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                      {"role": "user", "content": prompt}],
        )
        # Adjust response parsing for the chat completions format
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

app2 = gr.Interface(
    fn=generate_text_with_gpt,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question or comment..."),
    outputs="text",
    title="Food Explorer",
    description="This Food Explorer chatbot uses OpenAI's GPT-3.5 to generate responses. Enter any questions you have about this food, and let's explore and discover the deliciousness!",
    article = "Author: <a href=\"https://huggingface.co/ewanlong\">Ewan Long</a>",
)

# iface.launch()

pipe = pipeline(task="image-classification", 
                model="ewanlong/food_type_image_detection")
                
app1 = gr.Interface.from_pipeline(pipe, 
                           title="Food Recognition",
                           description="Food Categories Recognition",
                           examples=['apple_pie.jpg', 'dhokla.jpg'],
                           article = "Author: <a href=\"https://huggingface.co/ewanlong\">Ewan Long</a>",
                           )

demo = gr.TabbedInterface(
                          [app1, app2],
                          tab_names=["Food_Finder", "Food_Explorer"],
                          title="Food_Detection_with_Chatbot"
                          )

demo.launch()
