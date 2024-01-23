import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_table import DataTable
from dash import html, dcc, callback, Input, Output
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
dash.register_page(__name__)
prediction=pd.read_csv(r"E:\MLOPs_Webapp\prediction.csv")
# Initialize the Dash app
#app = dash.Dash(__name__)

# Define the app layout
layout = html.Div([
    html.H1("Data and Classification Report Viewer"),

    # DataTable to display first few rows of the DataFrame
    DataTable(
        id='data-table',
        columns=[{'name': col, 'id': col} for col in prediction.columns],
        #data=prediction.to_dict('records')

        data=prediction.head(5).to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto', 'overflowX': 'auto'},   # Add horizontal scroll bar
        style_cell={'maxWidth': 0}  #
    ),

    # Classification Report
    html.Div([
        html.H2("Classification Report"),
        html.Pre(
            children=classification_report(prediction["Actual"], prediction["prediction"]),
            style={'whiteSpace': 'pre-wrap'}
        )
    ])
])

# Run the app
# if __name__ == '__main__':
#     app.run_server(port=8060,debug=True)
