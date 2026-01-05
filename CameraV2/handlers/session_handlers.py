"""
Session management and feedback handlers
"""

import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

from handlers.state import sessions, openai_client
from services.ai_service import get_session_feedback as ai_get_session_feedback


async def get_session_feedback(exercise: str, reps_data: list, all_issues: list) -> str:
    """Get comprehensive feedback at session end. Uses OpenAI if available, otherwise rule-based."""
    
    if not reps_data:
        return "Henüz rep tamamlanmadı."
    
    # Try AI service first (uses OpenAI if available, otherwise rule-based)
    # ai_get_session_feedback is async, so we need to await it
    return await ai_get_session_feedback(exercise, reps_data, all_issues)


async def countdown_task(websocket: WebSocket, session_id: int):
    """Handle countdown after calibration: 3, 2, 1, START!"""
    session = sessions.get(session_id)
    if not session:
        return
    
    try:
        # State is already 'countdown' (set before this task was created)
        print("⏳ Countdown starting...")
        
        # Send calibration complete message
        try:
            await websocket.send_json({
                'type': 'state',
                'state': 'countdown',
                'message': 'Calibration complete! Get ready...'
            })
        except (RuntimeError, WebSocketDisconnect, AttributeError):
            return
        
        await asyncio.sleep(0.5)  # Brief pause before countdown
        
        # Countdown: 3, 2, 1
        for count in [3, 2, 1]:
            if websocket.client_state.name != 'CONNECTED':
                break
            print(f"⏳ Countdown: {count}")
            try:
                await websocket.send_json({
                    'type': 'countdown',
                    'number': count
                })
            except (RuntimeError, WebSocketDisconnect, AttributeError):
                return
            await asyncio.sleep(1)
        
        # START!
        if websocket.client_state.name == 'CONNECTED':
            try:
                await websocket.send_json({
                    'type': 'countdown',
                    'number': 0,  # 0 = START
                    'message': 'START!'
                })
                await asyncio.sleep(0.5)
                
                # Start tracking - THIS IS THE CRITICAL PART
                print("🏁 TRACKING STATE ACTIVATED!")
                session['state'] = 'tracking'
                session['tracking_frame_count'] = 0  # Reset frame count
                await websocket.send_json({
                    'type': 'state',
                    'state': 'tracking',
                    'message': 'Start exercising!'
                })
                print(f"✅ Session state is now: {session['state']}")
            except (RuntimeError, WebSocketDisconnect, AttributeError):
                pass
    except Exception as e:
        print(f"⚠️  Countdown error: {e}")
        import traceback
        traceback.print_exc()


async def rest_countdown_task(websocket: WebSocket, session_id: int, rest_time: int, next_set: int):
    """Handle rest countdown between sets."""
    session = sessions.get(session_id)
    if not session:
        return
    
    try:
        print(f"⏳ Rest countdown: {rest_time} seconds before set {next_set}")
        
        # Send rest start message
        try:
            await websocket.send_json({
                'type': 'rest_start',
                'rest_time': rest_time,
                'next_set': next_set
            })
        except (RuntimeError, WebSocketDisconnect, AttributeError):
            return
        
        # Countdown
        for remaining in range(rest_time, 0, -1):
            if websocket.client_state.name != 'CONNECTED':
                break
            try:
                await websocket.send_json({
                    'type': 'rest_countdown',
                    'remaining': remaining,
                    'next_set': next_set
                })
            except (RuntimeError, WebSocketDisconnect, AttributeError):
                return
            await asyncio.sleep(1)
        
        # Rest complete
        if websocket.client_state.name == 'CONNECTED':
            try:
                await websocket.send_json({
                    'type': 'rest_complete',
                    'next_set': next_set
                })
                session['state'] = 'tracking'
                session['current_set'] = next_set
                session['current_rep_in_set'] = 0
            except (RuntimeError, WebSocketDisconnect, AttributeError):
                pass
    except Exception as e:
        print(f"⚠️  Rest countdown error: {e}")

