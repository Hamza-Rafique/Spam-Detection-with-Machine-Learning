from src import train, predict

def main():
    choice = input("Enter 'train' to train the model or 'predict' to check a message: ")

    if choice == 'train':
        train.train_model('data/spam.csv')
    elif choice == 'predict':
        message = input("Enter the message to check: ")
        result = predict.predict_spam(message)
        print(f"Result: {result}")
    else:
        print("Invalid option. Please enter 'train' or 'predict'.")

if __name__ == "__main__":
    main()

