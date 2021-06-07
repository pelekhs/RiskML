"""
Main module of the server file
"""

# 3rd party modules
import connexion

# create the application instance
app = connexion.App(__name__, specification_dir="./")

# Read the swagger.yml file to configure the endpoints
app.add_api("swagger.yml")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5100, debug=True)
