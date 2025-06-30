from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = joblib.load('dataset.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['project_info']
    query_vec = vectorizer.transform([user_input])
    distances, indices = model.kneighbors(query_vec)

    results = df.iloc[indices[0]][[
        'Dataset_name', 'Author_name', 'Type_of_file',
        'Usability', 'Upvotes', 'Medals', 'Dataset_link'
    ]].to_dict(orient='records')

    return render_template('index.html', recommendations=results)

if __name__ == '__main__':
    app.run(debug=True)
