from views.layout import page_config
from views.sidebar import render_sidebar

from views.pages import dashboard
from views.pages import predict
from views.pages import predictions_history 
from views.pages import model_performance


PAGE_MAP = {
    "Dashboard": dashboard,
    "Predict Dropout": predict,
    "Predictions History": predictions_history, 
    "Model Performance": model_performance,
}

def main():
    page_config()
    active = render_sidebar()
    module = PAGE_MAP.get(active, dashboard)
    module.render()

if __name__ == "__main__":
    main()