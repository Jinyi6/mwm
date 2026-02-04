import http from 'k6/http'
import { check, sleep } from 'k6'

export const options = {
  vus: 10,
  duration: '5m',
}

const BASE = __ENV.BASE_URL || 'http://localhost:8000'

export default function () {
  const worldRes = http.post(`${BASE}/worlds/generate`, JSON.stringify({ seed_prompt: '', style: 'modern' }), {
    headers: { 'Content-Type': 'application/json' },
  })
  check(worldRes, { 'world created': (r) => r.status === 200 })
  const world = worldRes.json()

  const chatRes = http.post(`${BASE}/chats`, JSON.stringify({ world_id: world.id }), {
    headers: { 'Content-Type': 'application/json' },
  })
  check(chatRes, { 'chat created': (r) => r.status === 200 })
  const chat = chatRes.json()

  const msgRes = http.post(`${BASE}/chats/${chat.id}/message`, JSON.stringify({ role: 'user', content: '你好', desire_score: 6 }), {
    headers: { 'Content-Type': 'application/json' },
  })
  check(msgRes, { 'message sent': (r) => r.status === 200 })

  const autoRes = http.post(`${BASE}/chats/${chat.id}/auto/one`, null)
  check(autoRes, { 'auto one ok': (r) => r.status === 200 })

  sleep(1)
}
