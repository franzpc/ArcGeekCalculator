class SocialMedia:
    icons = {
        'twitter': 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABZUlEQVR4nG3TP0hWYRQG8J/fpxE6pBBUmxi0SVMQIk0R4SDR0OAiCE3ObSG4VFCrtYiTg0EFTS3i0hLUEkTpEkYoDlFDEQjZ1/A9t/tyvQde3nPPec6f55z7wjP0cJi7h0V9Gcw9gG6+t4JZi81ZfMUf3Mdb/MR4IxAeJfg9RuMHV4vKg/iN1/GdyD0fzDdciK1btrkSwCRmot+J7zJ+4S9my+CS30nsYDf29SS5hg/Rlxqz+S9VtksBriTxfqr28LIIHmgmKLMuJeAKpqPvYDiBnbbgkkoH7/A9+hP9oV5scm+TbhJNpPWnse/jE4YKTKtUNJbVP9XNVO/hYQPXGnwr4AO8SfvDuKeezTEq1cckfgR4A6eSYDP+j/iCkZJKJ2dUve8HRfK52G7jfPTVsuuq+vM4Xzn+BjbiO4OF6NdLCndj3MZp9b6rM5bWP+Ox/ob2cE64HoX7VMuAKn1WvZnq6b/4B/4LWuDRgVUDAAAAAElFTkSuQmCC',
        'youtube': 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABI0lEQVR4nJ3TPUoEQRAF4G/aFUUTTRUED+ANvIGIGCuCkQcw9SKGoicQL+EJzATxL1I2cf3fMZhqtm0m0YJiqK73uqreVDc628EulrGAecxiOvJfeMULhnjAWbgNtP/0TTiP4APjwr8rcJn7iLOLAVaizSk0JtYEOBWxAgsrCYs9gBb3QR5HXFrGLsJT0WIbgrU6UY+KEcqRMvaZTtly1s/47keVA7xVxOyjZPKrakvhx1jHVZyPC8wg1axKhzzrW4zWW6U3EeRvnRaXWCs42b6SbsNyxUyEEQ5xirlovamw73CjX6BrvxeoXqoWt0m32+Wt2VaLqk2Vy9hhwl0EfaubKlK5EzJ3q0r+xbdza3s6tZdMnvOMyY58hmD5OT/qnvLJD4DFjERqzxRYAAAAAElFTkSuQmCC',
        'facebook': 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABaUlEQVR4nGXTwUpVURQG4M97LxplDsU0SopeIBWJwFE+gQNfQWjUA/QEgThKE4LAmdBYx3IfogZJGJigdCHFuqX3ODhr5/K0YXP23uv//73Wv/ahHkOuxxw28Rm/Yn7Cu4hpcspiBKvoo4p5mdYV/mANtzK3jWHsBGiAiwb5BMdxXmE3LmyXLFYj0A+BQew3MIMJjGMliawV8lP8jcAgAbb8P+YjVvCzHbxEJ1IeillhHa3Yv8ASHiSxdnB9SbWX1C/wJIBj6DXMLLj9Fqayo+HDj8gIRtXu93DWwE4WQrm1wuswrbSqHanfw5sGtt/CYQAH8f2GoxAWmRzge/KgYI866GI6pbWMh3ir7v0YXqn7vtAooQvPXLevmFMlE+83DKwS9nlgbETgdwRP8SgZ1VP3/jJ5tlnILdzGXrrhDI+TwM8gF/O6uBPcf/WM4n0SyQLn6fwD7ja8uPE7L2I7lTCFr/iofpE3OFcz04JsI3W+UQAAAABJRU5ErkJggg==',
        'linkedin': 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABI0lEQVR4nK3Tvy5EURDH8c+9629EqRAqCpFsvWhVREciIt5BofIKEk/hMUS29ACi0VFrxJ9lF1dx53Jcf7IRk5xkvnNmJr9z5pwMx1hGz4c1/G5F5LSzgD9bjqfwz7GFo+DXPuq70AkVK0nT62jwEntF+CkX6OXIonATE9jGeMTz2CvCz5Pid+slwcuEu7iv7V3VFcBjwAFmsRa8iynsYQdDGMMGbuOIn+5gPxTNB6/XLmw08Q8jp5MnwcE440jwcPAkznCBOeX8TyMnSxsUIau6oDy4hSamsaicRKdqMKA/+/FNpA3ykNeocTW+euw9UNlDyLsJvgu+S3K+xFIFTaxiJnhBOeKlJKcVxa3g4l8+04mPR1Gt52/4WfnyusoP+Ir2G4qqYNvv83A0AAAAAElFTkSuQmCC',
        'tiktok': 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABS0lEQVR4nI3TP0scURQF8N/uTlRQWFKKgp0IlhZWmy8QrBKSSrC0tbDSzlJ7GxsLwW8hKKYIiLJ1Qj5BkkLBP7s7azH34XMdxQOHmXn33vPOve8Nr6OI5y6GweORmOYbAgmNeJb4hI/o4wNa7xFI6GMah5hCD4OiJrERzkbFmxhgBT+xj6s8oZXZzbGn6r/naRaJ98lBUocZzOIvfmWirdj5TjULGCuiuMRnbGAJbRxhNWulgUuso4NlTBdRvI2dEeuTNe2ktbOgAl+jeBA2f+McF5E8zATKZD29F9iKpCZO8AX/s6JcIGEQVGDe06C2ong8vnvBhBenVITSMILtbIcyOJc5ua4TOo1gH10sxnoTa6pje4ic7xFr5QKdKE6X41Y14W4mPMSPaK1R18o3/PPypiWeqv6D5OwZkp0FHOBP2L5RXZxNTNT1Do/R61gkb0Nc7QAAAABJRU5ErkJggg=='
    }

    social_links = '''<div style="text-align: right;">
        <div style="margin-bottom: 4px;">
            <span style="font-size: 11px;"><b>ArcGeek S.A.S. B.I.C.</b></span>
        </div>
        <div style="display: inline-block;">
            <a target="_blank" rel="noopener noreferrer" href="https://twitter.com/franzpc" style="text-decoration: none; margin: 0 3px;">
                <img title="Twitter" style="width: 16px; height: 16px; vertical-align: middle;" alt="twitter" src="data:image/png;base64,{twitter}">
            </a>
            <a target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/subscription_center?add_user=franzpc" style="text-decoration: none; margin: 0 3px;">
                <img title="YouTube" style="width: 16px; height: 16px; vertical-align: middle;" alt="youtube" src="data:image/png;base64,{youtube}">
            </a>
            <a target="_blank" rel="noopener noreferrer" href="https://www.facebook.com/arcgeek/" style="text-decoration: none; margin: 0 3px;">
                <img title="Facebook" style="width: 16px; height: 16px; vertical-align: middle;" alt="facebook" src="data:image/png;base64,{facebook}">
            </a>
            <a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/in/franzpc/" style="text-decoration: none; margin: 0 3px;">
                <img title="LinkedIn" style="width: 16px; height: 16px; vertical-align: middle;" alt="linkedin" src="data:image/png;base64,{linkedin}">
            </a>
            <a target="_blank" rel="noopener noreferrer" href="https://www.tiktok.com/@arcgeek" style="text-decoration: none; margin: 0 3px;">
                <img title="TikTok" style="width: 16px; height: 16px; vertical-align: middle;" alt="tiktok" src="data:image/png;base64,{tiktok}">
            </a>
            <a target="_blank" rel="noopener noreferrer" href="https://arcgeek.com/" style="text-decoration: none; margin-left: 5px;">
                <span style="font-size: 11px; color: black;"><b>www.arcgeek.com</b></span>
            </a>
        </div>
    </div>'''.format(**icons)