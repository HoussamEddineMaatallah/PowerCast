# Polyfill for Python 3.13+ missing imghdr module
# Streamlit 1.19 (installed on Py3.14) implicitly imports this

def what(file, h=None):
    return "png"
