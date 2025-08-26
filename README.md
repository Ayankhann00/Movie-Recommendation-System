# Movie-Recommendation-System
A user-based collaborative filtering movie recommendation system built with Python, leveraging the MovieLens 100K Dataset. The system recommends movies to users based on similarity scores computed from their rating patterns. The project includes a visualization of user similarities using Seaborn and a deployable web interface powered by Streamlit.

## Features
- Recommends top 5 movies for a given user based on ratings from similar users.
- Visualizes user similarity using a heatmap.
- Deployable web application where users can input their ID to get personalized recommendations.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For cosine similarity calculations.
- **Seaborn & Matplotlib**: For data visualization.
- **Streamlit**: For building the web application.

## Dataset
- **Source**: MovieLens 100K Dataset (included in the `ml-100k/` folder).
- **Details**: Contains 100,000 ratings from 1,000 users on 1,700 movies, provided in tab-delimited text files (`u.data`, `u.item`, `u.user`, etc.).
- **License**: Publicly available for research purposes from the GroupLens research lab.

## How to Run Locally

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/Ayankhann00/Movie-Recommendation-System/edit/main/README.md>
   Navigate to the project directory:
bashcd Movie-Recommendation-System

Install the required dependencies:
bashpip install -r requirements.txt

Run the Streamlit app:
bashstreamlit run app.py

This will open the app in your default web browser at http://localhost:8501.



Visualize the Similarity Matrix

Run the main script (e.g., main.py if you save the code with visualization) to generate and display a heatmap of user similarities.

Usage

Local App: Enter a user ID (between 1 and 943) in the Streamlit interface and click "Get Recommendations" to see the top 5 movie suggestions.
Visualization: The heatmap provides a visual representation of similarity scores between all users in the dataset.

Deployment

Platform: Deployed using Streamlit Community Cloud or Render.
Steps:

Push the repository to GitHub.
Connect your GitHub repository to the deployment platform.
The app will be live at the provided URL (e.g., https://your-app-url.streamlit.app).


Live URL: []

Project Structure
text /
│
├── ml-100k/                # MovieLens 100K Dataset files
│   ├── u.data
│   ├── u.item
│   ├── u.user
│   └── ...
│
├── app.py                 # Streamlit application file
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Optional: Files to exclude from Git
Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.
