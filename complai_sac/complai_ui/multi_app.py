"""Frameworks for running multiple Streamlit applications as a single app.
"""
import os
import platform
import streamlit as st

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.set_page_config(page_title='ComplAI', page_icon='🔎', layout='wide',
                           initial_sidebar_state='auto')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.title('ComplAI')

        st.write('''A Machine Learning Model Compliance Scanning App''')
        #models_list = os.listdir('/Users/AH00434/Documents/complai_examples')
        #current_path = os.path.abspath(os.getcwd())
        #complai_home = '/Users/AH00434/Documents/complai_examples'
        if(platform.system() == 'Windows'):
            complai_home = '\\'.join(os.path.abspath(os.getcwd()).split('\\')[:-1])
        elif(platform.system() == 'Darwin'):
            complai_home = '/'.join(os.path.abspath(os.getcwd()).split('/')[:-1])
        else:
            print('Unrecognized OS. Defaulting to Linux Based OS')
            complai_home = '/'.join(os.path.abspath(os.getcwd()).split('/')[:-1])

        models_list = os.listdir(complai_home)
        os.environ["complai_home"] = complai_home
        if('.DS_Store' in models_list):
            models_list.remove(('.DS_Store'))
        if('complai_ui' in models_list):
            models_list.remove(('complai_ui'))
        if ('README.md' in models_list):
            models_list.remove(('README.md'))
        if ('complai_scan-0.1.0.tar.gz' in models_list):
            models_list.remove(('complai_scan-0.1.0.tar.gz'))
        if('complai_scan-0.1.1-mlflow-integration.tar.gz' in models_list):
            models_list.remove(('complai_scan-0.1.1-mlflow-integration.tar.gz'))
        default_ix = 0
        if('lung_cancer_lr' in models_list):
            default_ix = models_list.index('lung_cancer_lr')
        model = st.sidebar.selectbox('Select the model for drift analysis', models_list, index=default_ix)
        st.sidebar.title('Navigation')
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])

        app['function'](model)