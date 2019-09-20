from app import app

######################
# main()
# Iniciamos Flask
######################
DEBUG = 1
PORT = 3333

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

