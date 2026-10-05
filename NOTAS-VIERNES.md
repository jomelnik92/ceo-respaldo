# Notas para cuando recupere créditos

Espacio donde el CEO agenda tareas y revisiones pendientes para retomarlas
el **viernes**, cuando se recuperen los créditos.

> Próximo viernes: **9 de octubre de 2026**

## Cómo usar este archivo

- Agrega cada pendiente como una casilla `- [ ]` en la sección que corresponda.
- Indica la prioridad: 🔴 alta · 🟡 media · 🟢 baja.
- Al completarla, márcala con `- [x]` y muévela a **Completado**.

---

## 🗓️ Tareas pendientes

- [ ] 

## 🔍 Revisiones pendientes

- [ ] 🔴 **Drop 289 (grupo Gatitos, 5/10/2026): salió mal y el bot lo dio por bueno.**
  Lo terminé a mano. Hay que averiguar qué causó la falla.
  Detalle en [Incidente: Drop 289](#incidente-drop-289).

## Incidente: Drop 289

**Qué reportó el bot (falso éxito):**

> 🧱 Drop 289 en Gatitos: 5 de 5 promos ajenas en tu muro (5 publicadas ahora)
> 🎥 video de verificacion mandado al grupo: muestra 5/5 promos
> ✅ Ya podés publicar lo tuyo: el drop terminó.

**Qué pasó en realidad (según el grupo):**

- 14:13: el bot manda el video de verificación.
- 14:14: otro asistente responde: *"En ningún video veo al host ni al que le sigue"*.
- 14:16 y 14:17: le pido 5 minutos.
- 14:22: ese asistente nos reporta: *"drop incompleto x3, no subió al host y le
  falta 1 modelo"*. El bot de moderación pasa el reporte a los admins.

**Lo que se vio en el grupo en medio del drop (13:49 a 13:56):**

- 13:49: aviso de llamada *"CUPOS DISPONIBLES, la llamada se detuvo"*.
- 13:50 y 13:51: el aviso *"lanzado la llamada. Durará 08 min 12 s"* aparece
  **dos veces**, a un minuto de distancia.
- 13:56: el owner silencia a un miembro con `/mute`; dos bots de moderación
  responden (uno con botones de Permisos/Unmute).

**Hipótesis a revisar:**

1. **Mensajes en medio de la lista** (tu sospecha principal): si el bot arma la
   lista del drop leyendo los mensajes del grupo en orden, los avisos de
   llamada, el `/mute` y las respuestas de los bots pudieron correr o cortar
   la lectura. Así se habrían salteado justo el host y el que le sigue.
   Revisar qué mensajes toma como "promo" y qué pasa si llega uno que no lo es.
2. **Llamada detenida y relanzada dos veces:** la secuencia detenida (13:49) →
   lanzada (13:50) → lanzada otra vez (13:51) puede haber reiniciado el estado
   del drop o creado dos drops a la vez. Revisar si el bot escucha esos avisos.
3. **La verificación no compara contra lo que pedía el grupo:** el bot
   confirmó "5 de 5" contra **su propio** conteo. Si la lista ya estaba mal
   desde el principio, el chequeo igual da OK. Lo esperado eran por lo menos
   7 (host + el que le sigue + 5), y faltaba otro modelo más.
4. **Otro factor:** límite de envío o error de Telegram al publicar el host
   (puede que haya fallado en silencio y el bot no lo contó), o que el host
   estuviera en el mensaje fijado y no en la lista.

**Qué necesito el viernes para revisarlo:**

- [ ] Acceso al código del bot. **No está en este repo**, que solo tiene la página web.
- [ ] Logs del bot del 5/10, entre 13:45 y 14:25.
- [ ] El mensaje original con la lista del drop 289 (host + modelos).
- [ ] Las capturas, que las tengo en el teléfono.
- [ ] Arreglo mínimo esperado: que el bot **no** diga "drop terminó" si no
      encontró al host, y que compare contra la lista completa del grupo.

## 💡 Ideas / para decidir

- 

---

## ✅ Completado

- 
