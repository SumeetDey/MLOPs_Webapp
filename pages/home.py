import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import pandas as pd
from dash import html, dcc, callback, Input, Output
dash.register_page(__name__, path='/')
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r"E:\Learning\Udemy\HR_Analytics.csv.csv")
categorical_columns = df.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)  # Features
y = df['Attrition'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
X_test["Actual"]=y_test
X_test["prediction"]=y_pred
X_test.to_csv(r"E:\MLOPs_Webapp\prediction.csv",index=False)




feature_importance = clf.feature_importances_

feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=True)

columns=list(feature_importance_df.Feature.tail(20))

# Initialize the Dash app
#app = dash.Dash(__name__)

# Define the app layout
layout = html.Div([
    html.H1("Histogram Viewer"),
    
    # Dropdown to select column
    dcc.Dropdown(
        id='column-dropdown_hist',
        options=[{'label': col, 'value': col} for col in columns],
        value=columns[0],  # Default selected column
        style={'width': '50%'}
    ),
    
   
    dcc.Graph(id='histogram-graph'),
    dcc.Dropdown(
        id='column-dropdown_box',
        options=[{'label': col, 'value': col} for col in columns],
        value=columns[0],  # Default selected column
        style={'width': '50%'}
    ),
    
   
    dcc.Graph(id='box-graph'),
])

# Define callback to update the graph based on column selection
@callback(
    Output('histogram-graph', 'figure'),
    [Input('column-dropdown_hist', 'value')]
)
def update_graph(selected_column):
    # Create histogram using Plotly Express
    
    fig = px.histogram(df, x=selected_column,color="Attrition", nbins=30)
    
    # Update layout
    fig.update_layout(
        title=f'Histogram for {selected_column}',
        xaxis_title=selected_column,
        yaxis_title='Frequency',
        bargap=0.1  # Set gap between bars
    )

    return fig



# Define callback to update the graph based on column selection
@callback(
    Output('box-graph', 'figure'),
    [Input('column-dropdown_box', 'value')]
)

def update_graph(selected_column):
    # Create box plot with points using Plotly Express
    fig = px.box(df, x=selected_column, points='all', color="Attrition")

    # Update layout
    fig.update_layout(
        title=f'Box Plot with Points for {selected_column}',
        xaxis_title=selected_column,
        yaxis_title='Values',
        boxmode='group'  # Display boxes in groups
    )

    return fig

# Run the app
# if __name__ == '__main__':
#     app.run_server(port=8040,debug=True)
