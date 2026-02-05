import numpy as np
import torch
import os, sys
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, project_root)
class DIB:
    def __init__(self, model, data, loss, name_process) -> None:
        self.model = model
        self.data = data
        self.loss = loss
        self.name_process = name_process

    def get_model(self):
        return self.model
    
    def get_train_loader(self):
        return self.data["train"]
    
    def get_test_loader(self):
        return self.data["test"]
    
    def train(self, optimizer, device, metrics):
        """
        Função responsável pelo treinamento do modelo neural multimodal

        Paramenters
        -----------
        optmizer (torch.optim): Otimizador otilizado no modelo neural
        device (str): Device onde serar computado os cálculos da rede

        Returns
        -----------
        total_loss, valor da soma das loss para cada época
        """
        for metric in metrics:
            metric.reset()

        self.model.train()
        losses = list()
        running_loss = 0.

        for batch_idx, (data, target) in enumerate(self.get_train_loader()):          
            ids =  data["ids"].to(device)
            mask= data["mask"].to(device)

            token_type_ids = data.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            targets_device = target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = self.model(ids, mask, token_type_ids)
                            
            loss = self.loss(outputs, targets_device)
            if isinstance(loss, (tuple, list)):
                loss = loss[0]
                
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            running_loss += loss.item()

            for metric in metrics:
                metric(outputs, targets_device)

            if batch_idx % 20 == 0 and batch_idx > 0:
                avg_loss = np.mean(losses_batch)
                message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(target), # len(target) é o batch_size real
                    len(self.get_train_loader().dataset),
                    100. * batch_idx / len(self.get_train_loader()), avg_loss
                )

                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses_batch = []
        total_loss = running_loss / len(self.get_train_loader())
        return total_loss, metrics
    
    def test(self, device, metrics):
        """
        Função responsável pela validação do treinamento do modelo.
        """
        with torch.no_grad():
            for metric in metrics:
                metric.reset()

            self.model.eval()
            running_loss = 0.
            
            loader = self.get_test_loader()

            for batch_idx, (data, target) in enumerate(loader):
                ids = data["ids"].to(device)
                mask = data["mask"].to(device)
                token_type_ids = data.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                targets_device = target.to(device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids)
                
                loss_outputs = self.loss(outputs, targets_device)

                loss = loss_outputs[0] if isinstance(loss_outputs, (tuple, list)) else loss_outputs
                running_loss += loss.item()

                for metric in metrics:
                    metric(outputs, targets_device)

        val_loss = running_loss / len(loader)
        
        return val_loss, metrics
    
    def fit(
        self, 
        optimizer, 
        scheduler, 
        num_epochs, 
        device,
        fold, 
        metrics=[], 
        start_epoch=0, 
        type_train="contrastive"
    ):
        """
        Orquestra o loop de épocas, alternando entre treino e validação.
        """
        print(f"Iniciando Treinamento: {num_epochs} épocas no dispositivo {device}")

        train_plot, val_plot = [], []

        for epoch in range(start_epoch, num_epochs):
            # 1. Estágio de Treinamento
            train_loss, metrics = self.train(optimizer, device, metrics)
            train_plot.append(train_loss)

            # 2. Atualização do Scheduler (Recomendado após o treino da época)
            if scheduler is not None:
                scheduler.step()

            # 3. Estágio de Validação
            val_loss, metrics = self.test(device, metrics)
            val_plot.append(val_loss)

            message = f"Epoch: {epoch + 1}/{num_epochs}"
            message += f"\n  Train Set: Average Loss: {train_loss:.4f}"
            for metric in metrics:
                message += f" | {metric.name()}: {metric.value()}"

            message += f"\n  Val Set:   Average Loss: {val_loss:.4f}"
            for metric in metrics:
                message += f" | {metric.name()}: {metric.value()}"

            print(message)
            print("-" * 30)

        return train_plot, val_plot