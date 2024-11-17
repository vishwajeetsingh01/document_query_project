import gradio as gr
from app import output_func,upload

# Main function for text input
def main(question):
    output = output_func(question)
    return output

# Function to handle PDF file upload
def handle_pdf(pdf_file):
    if pdf_file is not None:
        upload(pdf_file.name)
        file_name=pdf_file.name.split('\\')[-1]
        return f"PDF file '{file_name}' uploaded successfully!"
    else:
        return "No PDF file uploaded."

# Create Gradio interface
with gr.Blocks() as demo:
    # Add a title or heading
    gr.Markdown("# Document Query Project")

    # Section for PDF file upload
    pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
    pdf_output = gr.Textbox(label="PDF Upload Status")
    
    # Bind PDF upload function to file change
    pdf_upload.change(handle_pdf, inputs=pdf_upload, outputs=pdf_output)

    gr.Markdown("#### Get you answers from the document")
    # Section for text input, output, and submit button
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Enter your question", lines=4)
            submit_button = gr.Button("Submit")
        text_output = gr.Textbox(label="Output", lines=5)
    
    # Bind the main function to the submit button
    submit_button.click(main, inputs=text_input, outputs=text_output)
    
    

# Launch the app
demo.launch()
