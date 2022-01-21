# MRI Deep Learning
I hope to develop a deep learning model that classifies abnormalities (Specifically Alzheimer's Disease) in MRI scans. 
## Setup
Create AWS account, create user profile through IAM and create a bucket through S3. Upload MRI data to bucket and change the load_data function according to your bucket name and directories to file(s)

Create a virtual environment
```bash
virtualenv .venv
```
Download and install all required libraries
```bash
pip install -r requirements.txt
```
Create ".env" file and populate it with your AWS api keys within the following environment variables:
```bash
ACCESS_KEY=*API KEY GOES HERE EXCLUDING ASTERISKS*

SECRET_ACCESS_KEY=*SECRET API KEY GOES HERE EXCLUDING ASTERISKS*
```
Run the python file
```bash
python main.py
```