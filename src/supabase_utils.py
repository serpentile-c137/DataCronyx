import os
import bcrypt
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Initialize Supabase client
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def hash_password(password: str) -> str:
	"""Hash password using bcrypt"""
	salt = bcrypt.gensalt()
	hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
	return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
	"""Verify password against hash"""
	return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_user(username: str, password: str) -> bool:
	"""Create a new user account"""
	try:
		existing_user = supabase.table('users').select("id").eq('username', username).execute()
		if existing_user.data:
			return False  # User already exists
		hashed_password = hash_password(password)
		response = supabase.table('users').insert({
			'username': username,
			'password_hash': hashed_password
		}).execute()
		return len(response.data) > 0
	except Exception as e:
		print(f"Error creating user: {e}")
		return False

def authenticate(username: str, password: str) -> int:
	"""Authenticate user and return user ID if successful"""
	try:
		response = supabase.table('users').select("id, password_hash").eq('username', username).execute()
		if not response.data:
			return None  # User not found
		user = response.data[0]
		if verify_password(password, user['password_hash']):
			return user['id']
		return None  # Invalid password
	except Exception as e:
		print(f"Error authenticating user: {e}")
		return None

def get_user_by_id(user_id: int) -> dict:
	"""Get user information by ID"""
	try:
		response = supabase.table('users').select("id, username, created_at").eq('id', user_id).execute()
		if not response.data:
			return None
		return response.data[0]
	except Exception as e:
		print(f"Error getting user by ID: {e}")
		return None

def save_user_log(user_id: int, log_message: str) -> bool:
	"""Save a log message for a user"""
	try:
		response = supabase.table('logs').insert({
			'user_id': user_id,
			'log_message': log_message
		}).execute()
		return len(response.data) > 0
	except Exception as e:
		print(f"Error saving user log: {e}")
		return False

def get_user_logs(user_id: int) -> list:
	"""Retrieve all logs for a user"""
	try:
		response = supabase.table('logs').select("*").eq('user_id', user_id).order('created_at', desc=True).execute()
		return response.data if response.data else []
	except Exception as e:
		print(f"Error retrieving user logs: {e}")
		return []
