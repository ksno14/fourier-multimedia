import sys
import streamlit.web.cli as stcli

def main():
    """Programmatic entrypoint to run the Streamlit application."""
    sys.argv = ["streamlit", "run", "interface/streamlit_app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
