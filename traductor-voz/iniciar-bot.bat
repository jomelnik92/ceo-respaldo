@echo off
rem Arranca el bot de Telegram del traductor de voz.
rem Antes, una sola vez (en una terminal):
rem   setx TELEGRAM_TOKEN "el-token-que-te-dio-BotFather"
rem   setx TELEGRAM_USUARIOS "tu-numero-de-usuario"
cd /d "%~dp0"
call .venv\Scripts\activate.bat
traductor-voz bot
pause
