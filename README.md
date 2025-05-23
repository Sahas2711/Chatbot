
# LangGraph Chatbot

This is a chatbot application built with Streamlit. The chatbot interacts with users, providing responses based on pre-trained models or custom logic. It uses custom CSS for a personalized user interface, with dark-themed text and a clean layout.

## Features

- **Interactive Chatbot**: The chatbot engages in conversation with users and provides relevant responses.
- **Custom UI**: The interface is styled using custom CSS, ensuring a smooth and user-friendly experience.
- **Mobile Responsive**: The app is optimized for mobile devices with adaptive styling.

## Installation

### 1. Clone the Repository

To clone this project to your local machine, use the following command:

```bash
git clone https://github.com/Sahas2711/Chatbot.git
```

### 2. Navigate to the Project Directory

```bash
cd Chatbot
```

### 3. Install Dependencies

Make sure you have Python 3.x installed. Then, install the required packages by running:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, you can install Streamlit manually with:

```bash
pip install streamlit
```

### 4. Run the Streamlit App Locally

After installing dependencies, you can run the app locally using:

```bash
streamlit run main.py
```

This will launch the chatbot in your browser at `http://localhost:8501`.

## Deployment on Streamlit Cloud

If you prefer to deploy the app on Streamlit Cloud, follow these steps:

1. **Go to Streamlit Cloud**: Visit [Streamlit Cloud](https://share.streamlit.io/).
2. **Sign In**: Log in or create an account if you don't have one.
3. **Link to GitHub Repository**:

   * Click on "New App" and select the GitHub repository where your code is located.
   * Streamlit will automatically deploy the app for you, and any updates you push to GitHub will trigger an automatic redeployment.

You can access your deployed app through the URL provided after deployment. Example:

```bash
https://sahas2711-chatbot-main-pmm0jk.streamlit.app/
```

## Customizations

* **CSS Styling**: The custom CSS is used for UI styling. You can modify the colors, padding, font sizes, and layout by adjusting the CSS inside the `main.py` file.
* **App Title**: The page title is set to "LangGraph Chatbot", but you can change this by modifying the `st.set_page_config` in the code.

## Folder Structure

```
.
├── main.py            # The main Streamlit app file
├── requirements.txt    # List of Python dependencies
├── .env                # (Optional) Store environment variables like API keys
└── README.md           # This file
```

## Contributing

Feel free to fork this repository, make changes, and create a pull request. If you encounter any issues or have suggestions, open an issue in the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
