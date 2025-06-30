:: This is a special file that helps run your website automatically!
:: When you double-click this file, it will start your website for you

@echo off
:: This line above makes the screen less messy by hiding some technical stuff

:: This is like putting a bookmark in your favorite book
:: When the computer sees this, it remembers where to come back to
:start

:: This line shows a friendly message to tell you what's happening
echo Starting Django Development Server...

:: This is the magic line that starts your website!
:: It tells Python (the programming language) to run your Django website
python manage.py runserver

:: If the website stops for any reason, this message will appear
echo Server stopped, restarting in 3 seconds...

:: This line tells the computer to wait for 3 seconds
:: It's like counting "1... 2... 3..." before starting again
:: The /nobreak means it won't show the counting
:: > nul means it won't show extra messages
timeout /t 3 /nobreak > nul

:: This line tells the computer to go back to the bookmark we made
:: It's like saying "do it all over again!"
:: This makes sure your website keeps running even if it stops
goto start

:: How to use this file:
:: 1. Double-click this file to start your website
:: 2. Your website will be available at: http://127.0.0.1:8000/
:: 3. To stop the website, press Ctrl+C two times
:: 4. If the website crashes, it will restart automatically after 3 seconds! 