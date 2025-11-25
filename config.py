# Author: Yifan Wang
class Config:
    START_DATE = '1999-01-01'
    END_DATE = None
    SEQ_LEN = 60
    DATA_DIR = 'data/'
    TARGET_SYMBOL = 'QQQ'
    TRAIN_SYMBOLS = [
        'QQQ', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 
        'ADBE', 'CSCO', 'INTC', 'CMCSA', 'PEP', 'NFLX', 'TXN', 
        'AVGO', 'GILD', 'COST', 'QCOM'
    ]
    INPUT_DIM = 2
    LATENT_DIM = 32
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    MODEL_PATH = 'checkpoints/best_model.pth'
