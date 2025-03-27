# Create doc_rag/api/schemas.py

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class MessageBase(BaseModel):
    content: str
    is_user: bool = True

class MessageCreate(MessageBase):
    pass

class Message(MessageBase):
    id: str
    conversation_id: str
    timestamp: datetime
    query_metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True

class ConversationBase(BaseModel):
    title: str

class ConversationCreate(ConversationBase):
    pass

class Conversation(ConversationBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []
    
    class Config:
        orm_mode = True