import argparse
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Preparation
class CLEVRDataset(Dataset):
    def __init__(self, image_dir, question_file, tokenizer, max_len, answer_to_id):
        self.image_dir = image_dir
        logger.info(f"Loading questions from {question_file}")
        start_time = time.time()
        with open(question_file, 'r') as f:
            self.questions = json.load(f)['questions']
        logger.info(f"Loaded {len(self.questions)} questions in {time.time() - start_time:.2f} seconds")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.answer_to_id = answer_to_id
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Reduced image size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Pre-check image existence
        self.valid_indices = []
        for idx, q in enumerate(self.questions):
            image_path = os.path.join(self.image_dir, q['image_filename'])
            if os.path.exists(image_path):
                self.valid_indices.append(idx)
            else:
                logger.warning(f"Image not found: {image_path}")
        logger.info(f"Found {len(self.valid_indices)} valid image-question pairs out of {len(self.questions)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        question_data = self.questions[actual_idx]
        image_path = os.path.join(self.image_dir, question_data['image_filename'])
        start_time = time.time()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        logger.debug(f"Loaded image {image_path} in {time.time() - start_time:.4f} seconds")

        question = question_data['question']
        tokens = self.tokenizer.encode(question, add_special_tokens=True)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))

        attention_mask = [1 if token != 0 else 0 for token in tokens]
        answer = self.answer_to_id[question_data['answer']]

        return {
            'image': image,
            'question': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'answer': torch.tensor(answer, dtype=torch.long)
        }

# Build answer vocabulary
def build_answer_vocab(question_file):
    logger.info(f"Building answer vocabulary from {question_file}")
    start_time = time.time()
    with open(question_file, 'r') as f:
        questions = json.load(f)['questions']
    answers = [q['answer'] for q in questions]
    # unique_answers = list(set(answers))
    # print(f"Unique answers: {unique_answers}")
    unique_answers = sorted(list(set(answers)))
    print(f"Unique answers: {unique_answers}")
    logger.info(f"Built vocabulary with {len(unique_answers)} unique answers in {time.time() - start_time:.2f} seconds")
    return {ans: idx for idx, ans in enumerate(unique_answers)}
# Model Architecture
class ImageEncoder(nn.Module):
    def __init__(self, dv_embed=768):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Linear(2048, dv_embed)
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("Initialized ImageEncoder with frozen ResNet101")

    def forward(self, x):
        start_time = time.time()
        x = self.features(x)  # [B, 2048, h, w]
        x = x.permute(0, 2, 3, 1)  # [B, h, w, 2048]
        x = x.view(x.size(0), -1, 2048)  # [B, h*w, 2048]
        x = self.projection(x)  # [B, h*w, 768]
        logger.debug(f"ImageEncoder forward pass took {time.time() - start_time:.4f} seconds")
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, max_len=50, num_layers=6, nhead=8):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        logger.info("Initialized TextEncoder with Transformer")

    def forward(self, tokens, attention_mask):
        start_time = time.time()
        x = self.embedding(tokens) + self.positional_embedding[:, :tokens.size(1), :]
        output = self.transformer_encoder(x, src_key_padding_mask=(attention_mask == 0))
        logger.debug(f"TextEncoder forward pass took {time.time() - start_time:.4f} seconds")
        return output

class FeatureFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(FeatureFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        logger.info("Initialized FeatureFusion with Cross-Attention")

    def forward(self, text_features, image_features):
        start_time = time.time()
        output, _ = self.cross_attention(text_features, image_features, image_features)
        logger.debug(f"FeatureFusion forward pass took {time.time() - start_time:.4f} seconds")
        return output

class Decoder(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=500, num_classes=None):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        logger.info("Initialized Decoder")

    def forward(self, x):
        start_time = time.time()
        output = self.mlp(x)
        logger.debug(f"Decoder forward pass took {time.time() - start_time:.4f} seconds")
        return output

class VQAModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=768, max_len=50):
        super(VQAModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, max_len)
        self.feature_fusion = FeatureFusion(embed_dim)
        self.decoder = Decoder(embed_dim, num_classes=num_classes)
        logger.info("Initialized VQAModel")

    def forward(self, image, tokens, attention_mask):
        start_time = time.time()
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(tokens, attention_mask)
        fused_features = self.feature_fusion(text_features, image_features)
        cls_feature = fused_features[:, 0, :]
        output = self.decoder(cls_feature)
        logger.debug(f"VQAModel forward pass took {time.time() - start_time:.4f} seconds")
        return output
    
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def compute_accuracy(preds, targets):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    scaler = GradScaler()  # For mixed precision training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        logger.info(f"Starting training epoch {epoch+1}/{num_epochs}")
        train_start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch_start_time = time.time()
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            masks = batch['attention_mask'].to(device)
            answers = batch['answer'].to(device)
            logger.debug(f"Batch {batch_idx} data transfer to GPU took {time.time() - batch_start_time:.4f} seconds")
            
            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(images, questions, masks)
                loss = criterion(outputs, answers)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_acc += compute_accuracy(outputs, answers)
            logger.debug(f"Batch {batch_idx} processing took {time.time() - batch_start_time:.4f} seconds")
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        logger.info(f"Epoch {epoch+1} training took {time.time() - train_start_time:.2f} seconds")
        
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        logger.info(f"Starting validation for epoch {epoch+1}")
        val_start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                questions = batch['question'].to(device)
                masks = batch['attention_mask'].to(device)
                answers = batch['answer'].to(device)
                with autocast():
                    outputs = model(images, questions, masks)
                    loss = criterion(outputs, answers)
                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, answers)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        logger.info(f"Epoch {epoch+1} validation took {time.time() - val_start_time:.2f} seconds")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_save_path = f"/kaggle/working/focalnbert_vqa_model_e{epoch+1}_vacc{val_acc:.4f}_{timestamp}.pth"
        torch.save(model.state_dict(), model_save_path)
        # torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_test(model, test_loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    logger.info("Starting test evaluation")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            masks = batch['attention_mask'].to(device)
            answers = batch['answer'].to(device)
            with autocast():
                outputs = model(images, questions, masks)
                _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(answers.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1, all_preds, all_targets

def visualize_predictions(dataset, model, tokenizer, answer_to_id, device, num_samples=5, errors_only=False, save_prefix=''):
    model.eval()
    id_to_answer = {v: k for k, v in answer_to_id.items()}
    indices = np.random.choice(len(dataset), len(dataset), replace=False)
    samples_shown = 0
    
    logger.info(f"Visualizing {'errors' if errors_only else 'predictions'}")
    for idx in indices:
        if samples_shown >= num_samples:
            break
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        question = sample['question'].unsqueeze(0).to(device)
        mask = sample['attention_mask'].unsqueeze(0).to(device)
        answer = sample['answer'].item()
        
        with torch.no_grad():
            with autocast():
                output = model(image, question, mask)
                _, predicted = torch.max(output, 1)
                predicted_answer = predicted.item()
        
        if errors_only and predicted_answer == answer:
            continue
        
        image_np = sample['image'].permute(1, 2, 0).numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image_np = image_np.astype(np.uint8)
        question_text = tokenizer.decode(sample['question'].tolist(), skip_special_tokens=True)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image_np)
        plt.title(f"Q: {question_text}\nPred: {id_to_answer[predicted_answer]}, GT: {id_to_answer[answer]}")
        plt.axis('off')
        plt.savefig(f'prediction_{save_prefix}_{samples_shown+1}{"_error" if errors_only else ""}.png')
        plt.close()
        samples_shown += 1

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


def main():
    parser = argparse.ArgumentParser(description="Train or run inference on a model.")

    # Define the command-line arguments
    parser.add_argument('--mode', type=str, required=True, help='Mode of operation: train or inference')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_path', type=str, help='Path to save the model (only used in training mode)')
    parser.add_argument('--model_path', type=str, help='Path to the model (only used in inference mode)')
    args = parser.parse_args()

    # Check the mode of operation
    if args.mode == 'train':
        print(f"Training the model using dataset: {args.dataset}")
        print(f"Model will be saved to: {args.save_path}")
        print(f"Using model: {args.model_path}")
        # Call the training function
        train_model(args.dataset, args.save_path, args.model_path)
    
    elif args.mode == 'inference':
        print(f"Running inference using dataset: {args.dataset}")
        print(f"Using model: {args.model_path}")
        
        # Call the inference function
        run_inference(args.dataset, args.model_path)
    
    else:
        print("Invalid mode. Please use '--mode train' for training or '--mode inference' for inference.")
        
def train_model(dataset_path, save_path, model_path):
    # Placeholder for training logic
    print(f"Training model on dataset from {dataset_path}")
    print(f"Saving model to {save_path}")
    print(f"Using model from {model_path}")
    # Main Execution - Setup and Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 50
    print("1")

    # Updated paths based on your directory structure
    base_path = dataset_path
    base_path = os.path.join(base_path, 'CLEVR_COL774_A4')
    train_img_dir = os.path.join(base_path, 'images/trainA')
    val_img_dir = os.path.join(base_path, 'images/valA')
    test_img_dir = os.path.join(base_path, 'images/testA')
    train_q_file = os.path.join(base_path, 'questions/CLEVR_trainA_questions.json')
    val_q_file = os.path.join(base_path, 'questions/CLEVR_valA_questions.json')
    test_q_file = os.path.join(base_path, 'questions/CLEVR_testA_questions.json')
    print("2")

    answer_to_id = build_answer_vocab(train_q_file)
    if os.path.isfile(save_path):
        answer_vocab_path = os.path.join(os.path.dirname(save_path), 'answer_to_id.json')
    else:
        answer_vocab_path = os.path.join(save_path, 'answer_to_id.json')

    with open(answer_vocab_path, 'w') as f:
        json.dump(answer_to_id, f)
    num_classes = len(answer_to_id)
    print("3")

    train_dataset = CLEVRDataset(train_img_dir, train_q_file, tokenizer, max_len, answer_to_id)
    val_dataset = CLEVRDataset(val_img_dir, val_q_file, tokenizer, max_len, answer_to_id)
    print("4")

    batch_size = 32  # Reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("5")

    model = VQAModel(tokenizer.vocab_size, num_classes, max_len=max_len).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    logger.info("Initialized Focal Loss with alpha=1, gamma=2")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    resnet_params = list(model.image_encoder.features.parameters())
    other_params = [param for name, param in model.named_parameters() if "image_encoder.features" not in name]
    optimizer = torch.optim.Adam([
        {'params': resnet_params, 'lr': 1e-6, 'weight_decay': 1e-4},
        {'params': other_params, 'lr': 1e-5, 'weight_decay': 1e-4}
    ])
    logger.info("Initialized optimizer with differential learning rates for Focal Loss training")

    # Train with Focal Loss
    num_focal_epochs = 5  # Reduced epochs to prevent overfitting
    logger.info(f"Starting training with Focal Loss for {num_focal_epochs} epochs")
    focal_train_losses, focal_val_losses, focal_train_accs, focal_val_accs = train_and_evaluate(
        model, train_loader, val_loader, focal_criterion, optimizer, num_focal_epochs, device
    )
    print("6")

    # Save the model
    if not os.path.basename(save_path):
        save_path = os.path.join(save_path, "vqa_model_focal.pth")
    model_save_path = save_path
    # torch.save(model, model_save_path)  # Save the entire model
    torch.save(model.state_dict(), model_save_path)
    print(f'focal trained model saved at: {model_save_path}')



def run_inference(dataset_path, model_path):
    print(f"Running inference on dataset from {dataset_path}")
    print(f"Using model from {model_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 50
    print("1")
    # Define dataset paths
    base_path = dataset_path
    base_path = os.path.join(base_path, 'CLEVR_COL774_A4')
    test_img_dir = os.path.join(base_path, 'images/testA')
    test_q_file = os.path.join(base_path, 'questions/CLEVR_testA_questions.json')
    print("2")
    # Load answer vocabulary
    if not os.path.basename(model_path):
        model_path = os.path.join(model_path, "vqa_model.pth")
    logger.info(f"Final model path: {model_path}")
    # Load answer vocabulary
    if os.path.isfile(model_path):
        answer_vocab_path = os.path.join(os.path.dirname(model_path), 'answer_to_id.json')
    else:
        answer_vocab_path = os.path.join(model_path, 'answer_to_id.json')
    
    if os.path.isfile(answer_vocab_path):
        with open(answer_vocab_path, 'r') as f:
            answer_to_id = json.load(f)
    else:
        logger.warning(f"Answer vocabulary file not found at {answer_vocab_path}. Recomputing answer_to_id.")
        train_q_file = os.path.join(base_path, 'questions/CLEVR_trainA_questions.json')
        if os.path.isfile(train_q_file):
            logger.info("Recomputing answer_to_id using train data.")
            answer_to_id = build_answer_vocab(train_q_file)
        else:
            logger.warning("Train data not found. Recomputing answer_to_id using test data.")
            test_q_file = os.path.join(base_path, 'questions/CLEVR_testA_questions.json')
            answer_to_id = build_answer_vocab(test_q_file)
    num_classes = len(answer_to_id)

    # Initialize test dataset and dataloader
    test_dataset = CLEVRDataset(test_img_dir, test_q_file, tokenizer, max_len, answer_to_id)
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("3")
    # Initialize and load model
    model = VQAModel(tokenizer.vocab_size, num_classes, max_len=max_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  
    logger.info(f"Loaded model from {model_path}")

    # Define criterion (only needed if computing loss in evaluate_test)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    # Run evaluation
    accuracy, precision, recall, f1, all_preds, all_targets = evaluate_test(model, test_loader, focal_criterion, device)

    # Convert predictions to answers
    id_to_answer = {v: k for k, v in answer_to_id.items()}
    predicted_answers = [id_to_answer[pred] for pred in all_preds]

    # Log results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Generated {len(predicted_answers)} predictions")

    # Save results
    # results = {
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'predictions': predicted_answers,
    #     'targets': [id_to_answer[target] for target in all_targets]
    # }
    # with open('inference_results.json', 'w') as f:
    #     json.dump(results, f)
    # logger.info("Inference and evaluation completed. Results saved to inference_results.json")

    # return results


if __name__ == "__main__":
    main()
