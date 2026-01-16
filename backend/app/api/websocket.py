"""
WebSocket handler for real-time event streaming
WS /ws/events - Real-time ADL event notifications
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set
import json
import asyncio

router = APIRouter()

# Connection manager (pseudo-implementation)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # Clean up dead connections
        self.active_connections -= disconnected

manager = ConnectionManager()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming
    
    Message format sent to client:
        {
            "type": "adl_event",
            "data": {
                "event_id": 123,
                "label": "fall",
                "severity": "critical",
                "camera_id": 1,
                "confidence": 0.95,
                "timestamp": "2026-01-16T10:30:00Z"
            }
        }
    
    Usage (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/events');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('New ADL event:', data);
        };
    """
    await manager.connect(websocket)
    
    try:
        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()
            # Could handle client commands here (e.g., subscribe to specific cameras)
            # For now, just echo back
            await websocket.send_json({
                "type": "ack",
                "message": f"Received: {data}"
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Utility function to be called by pipeline when new event detected
async def notify_new_event(event_data: dict):
    """
    Called by PipelineService when a new ADL event is detected
    Broadcasts to all connected WebSocket clients
    """
    await manager.broadcast({
        "type": "adl_event",
        "data": event_data
    })
