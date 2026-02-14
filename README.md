# mm-rag-test
⚠️ Critical Security Warning (The "Gotcha")
DO NOT save your API keys in the code before pushing to GitHub.
If you save os.environ["GOOGLE_API_KEY"] = "AIzaSy..." and push it, bots will steal your key in seconds.

The Fix:

In the code, delete your actual key and leave the input prompt:

Python
# BAD:
# api_key = "AIzaSy_SECRET_KEY" 

# GOOD: 
import getpass
api_key = getpass.getpass("Enter Key:")
Tell your group mates to generate their own free Google Gemini keys to run the notebook. This is safer and professional.
