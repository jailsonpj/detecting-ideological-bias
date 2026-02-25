import os
import sys
import copy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from transformers import logging

logging.set_verbosity_warning()
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.utils.utils import read_json
from src.utils.early_stopping import EarlyStopping
from src.networks.networks import TransformerNet
from src.dib.dib import DIB
from src.losses.losses import OnlineTripletLoss, OnlineContrastiveLoss, ContrastiveLoss
from src.selectors.selectors import SemihardNegativeTripletSelector, AllPositivePairSelector
from src.preprocess.dataset_custom import NewsPaperData, CustomDataset

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


class RunDIB:
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def read_data(self):
        # Instanciar leitores
        dataset_train = NewsPaperData(
            self.config["dataset"]["path_base"],
            self.config["dataset"]["documents_train"]
        )
        dataset_val = NewsPaperData(
            self.config["dataset"]["path_base"],
            self.config["dataset"]["documents_val"]
        )

        # Obter DataFrames
        df_train, _ = dataset_train.get_dataset_custom( # Adicionado parâmetro faltante
            dict_rename=self.config["dataset"]["rename_columns"],
            corpus=self.config["dataset"]["corpus"]
        )
        df_val, _ = dataset_val.get_dataset_custom(
            dict_rename=self.config["dataset"]["rename_columns"],
            corpus=self.config["dataset"]["corpus"]
        )

        df_train = df_train.dropna(subset=["text"]).reset_index(drop=True)
        df_val = df_val.dropna(subset=["text"]).reset_index(drop=True)

        df_train = df_train.dropna(subset=["labels"]).reset_index(drop=True)
        df_val = df_val.dropna(subset=["labels"]).reset_index(drop=True)

        return df_train, df_val

    def get_loss(self, loss_name):
        margin = 1.0
        loss_map = {
            "contrastive": lambda: OnlineContrastiveLoss(
                margin=margin,
                pair_selector=AllPositivePairSelector()
            ),
            "cross": lambda: torch.nn.CrossEntropyLoss(),
            "triplet-semihard": lambda: OnlineTripletLoss(
                margin=margin, 
                triplet_selector=SemihardNegativeTripletSelector(margin=margin)
            )
        }
        if loss_name not in loss_map:
            raise ValueError(f"Loss '{loss_name}' não reconhecida.")
        return loss_map[loss_name]()

    def execute(self):   
        df_train, _ = self.read_data()
        tokenizer = AutoTokenizer.from_pretrained(self.config["pre_trained"])

        ds_train = CustomDataset(
            df=df_train,
            tokenizer=tokenizer,
            max_len=512
        )

        loss_func = self.get_loss(self.config["function_loss"]["loss"])
        
        kfold = KFold(n_splits=self.config["train"]["folds"], shuffle=True, random_state=42)
        
        best_avg_loss = float('inf')
        save_model = None
        save_fold = None

        for fold, (train_ids, test_ids) in enumerate(kfold.split(ds_train)):
            print(f"\n-------- Fold: {fold + 1} --------")
            
            # Samplers e Loaders
            train_loader = DataLoader(ds_train, batch_size=10, sampler=SubsetRandomSampler(train_ids))
            test_loader = DataLoader(ds_train, batch_size=10, sampler=SubsetRandomSampler(test_ids))

            # Inicialização do Modelo a cada Fold (Garante independência)
            model = TransformerNet(name_model=self.config["pre_trained"]).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
            fold_ckpt_path = f"{self.config['path_base_save_model']}/temp_fold_{fold}.pt"
            early_stopping = EarlyStopping(patience=5, verbose=True, path=fold_ckpt_path)

            dib = DIB(
                model=model,
                data={"train": train_loader, "test": test_loader},
                loss=loss_func,
                name_process=self.config["name_process"]
            )

            train_plot, val_plot, fold_best_loss = dib.fit(
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=self.config["train"]["num_epochs"],
                device=self.device,
                fold=fold,
                early_stopping=early_stopping,
                metrics=[],
                type_train=self.config["name_process"]
            )

            if fold_best_loss < best_avg_loss:
                best_avg_loss = fold_best_loss
                save_fold = fold
                save_model = copy.deepcopy(dib.get_model().state_dict())
                print(f"Novo melhor modelo encontrado no Fold {fold} com Loss: {best_avg_loss:.4f}")

        if save_model is not None:
            save_path = os.path.join(self.config["path_base_save_model"], self.config["name_process"])
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, f"{self.config['name_process']}_fold{save_fold}_model.pth")
            
            torch.save(save_model, full_path) 
            print(f"\nTop 1 Median Loss: {best_avg_loss:.4f} - Salvo em: {full_path}")

def run():
    file_json = sys.argv[1]
    config = read_json(file_json)
    run_dib = RunDIB(config=config)
    run_dib.execute()

if __name__ == "__main__":
    run()