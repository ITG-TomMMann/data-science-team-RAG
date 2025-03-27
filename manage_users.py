import sys
import os
import argparse
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doc_rag.database.database import SessionLocal
from doc_rag.users.users_model import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    """Hash a password for storing."""
    return pwd_context.hash(password)

def create_user(username, email, password, is_admin=False):
    """Create a new user in the database."""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            print(f"User {username} already exists.")
            return False
        
        # Check if email already exists
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            print(f"Email {email} already exists.")
            return False
        
        # Hash the password
        hashed_password = get_password_hash(password)
        
        # Create the user
        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_active=True
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        print(f"User {username} created successfully!")
        return True
    except Exception as e:
        db.rollback()
        print(f"Error creating user: {str(e)}")
        return False
    finally:
        db.close()

def update_user(username, new_username=None, new_password=None, new_email=None):
    """Update an existing user's details."""
    db = SessionLocal()
    try:
        # Find the user
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"User {username} not found.")
            return False
        
        # Update username if provided
        if new_username and new_username != username:
            # Check if new username exists
            existing = db.query(User).filter(User.username == new_username).first()
            if existing:
                print(f"Username {new_username} already exists.")
                return False
            user.username = new_username
        
        # Update email if provided
        if new_email and new_email != user.email:
            # Check if new email exists
            existing = db.query(User).filter(User.email == new_email).first()
            if existing:
                print(f"Email {new_email} already exists.")
                return False
            user.email = new_email
        
        # Update password if provided
        if new_password:
            user.hashed_password = get_password_hash(new_password)
        
        db.commit()
        
        print(f"User updated successfully!")
        return True
    except Exception as e:
        db.rollback()
        print(f"Error updating user: {str(e)}")
        return False
    finally:
        db.close()

def list_users():
    """List all users in the database."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("No users found.")
            return
        
        print("\nUsers in the database:")
        print("-" * 60)
        print(f"{'ID':<36} | {'Username':<20} | {'Email':<30}")
        print("-" * 60)
        
        for user in users:
            print(f"{str(user.id):<36} | {user.username:<20} | {user.email:<30}")
        
        print("-" * 60)
        print(f"Total: {len(users)} users")
    except Exception as e:
        print(f"Error listing users: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User management tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create user parser
    create_parser = subparsers.add_parser("create", help="Create a new user")
    create_parser.add_argument("--username", "-u", required=True, help="Username")
    create_parser.add_argument("--email", "-e", required=True, help="Email address")
    create_parser.add_argument("--password", "-p", required=True, help="Password")
    create_parser.add_argument("--admin", "-a", action="store_true", help="Create as admin user")
    
    # Update user parser
    update_parser = subparsers.add_parser("update", help="Update an existing user")
    update_parser.add_argument("--username", "-u", required=True, help="Current username")
    update_parser.add_argument("--new-username", "-n", help="New username")
    update_parser.add_argument("--new-email", "-e", help="New email address")
    update_parser.add_argument("--new-password", "-p", help="New password")
    
    # List users parser
    subparsers.add_parser("list", help="List all users")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_user(args.username, args.email, args.password, args.admin)
    elif args.command == "update":
        if not any([args.new_username, args.new_email, args.new_password]):
            print("Error: At least one of --new-username, --new-email, or --new-password must be provided")
        else:
            update_user(args.username, args.new_username, args.new_password, args.new_email)
    elif args.command == "list":
        list_users()
    else:
        parser.print_help()