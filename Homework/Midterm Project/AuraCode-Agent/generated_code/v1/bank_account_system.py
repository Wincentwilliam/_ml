import datetime
from typing import Dict, List

class BankAccount:
    def __init__(self, account_number: str, owner_name: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self.owner_name = owner_name
        self.balance = initial_balance
        self.transactions: List[Dict] = []
    
    def deposit(self, amount: float) -> bool:
        if amount <= 0:
            print("Error: Deposit amount must be positive.")
            return False
        self.balance += amount
        self.transactions.append({
            'type': 'DEPOSIT',
            'amount': amount,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        print(f"Deposited ${amount:.2f}. New balance: ${self.balance:.2f}")
        return True
    
    def withdraw(self, amount: float) -> bool:
        if amount <= 0:
            print("Error: Withdrawal amount must be positive.")
            return False
        if amount > self.balance:
            print(f"Error: Insufficient funds. Current balance: ${self.balance:.2f}")
            return False
        self.balance -= amount
        self.transactions.append({
            'type': 'WITHDRAWAL',
            'amount': amount,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        print(f"Withdrew ${amount:.2f}. New balance: ${self.balance:.2f}")
        return True
    
    def transfer(self, target_account: 'BankAccount', amount: float) -> bool:
        if amount <= 0:
            print("Error: Transfer amount must be positive.")
            return False
        if amount > self.balance:
            print(f"Error: Insufficient funds. Current balance: ${self.balance:.2f}")
            return False
        self.balance -= amount
        target_account.balance += amount
        self.transactions.append({
            'type': 'TRANSFER_OUT',
            'amount': amount,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        target_account.transactions.append({
            'type': 'TRANSFER_IN',
            'amount': amount,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        print(f"Transferred ${amount:.2f} to account {target_account.account_number}.")
        return True
    
    def view_history(self) -> None:
        if not self.transactions:
            print("No transactions found.")
            return
        print(f"\nTransaction History for Account {self.account_number} ({self.owner_name}):")
        print("-" * 60)
        for tx in self.transactions:
            print(f"{tx['timestamp']} | {tx['type']:12} | ${tx['amount']:>10.2f}")
        print("-" * 60)
        print(f"Current Balance: ${self.balance:.2f}")
        print(f"Total Transactions: {len(self.transactions)}")

class BankSystem:
    def __init__(self):
        self.accounts: Dict[str, BankAccount] = {}
        self.next_account_number = 1000
    
    def create_account(self) -> BankAccount:
        owner = input("Enter owner name: ").strip()
        if not owner:
            print("Error: Owner name cannot be empty.")
            return None
        account_num = f"ACC{self.next_account_number:04d}"
        self.next_account_number += 1
        account = BankAccount(account_num, owner)
        self.accounts[account_num] = account
        print(f"Account created successfully: {account_num}")
        return account
    
    def list_accounts(self) -> None:
        if not self.accounts:
            print("No accounts found.")
            return
        print("\nAvailable Accounts:")
        print("-" * 40)
        for acc_num, acc in self.accounts.items():
            print(f"{acc_num} | {acc.owner_name} | Balance: ${acc.balance:.2f}")
        print("-" * 40)
    
    def view_account(self, account_number: str) -> BankAccount:
        if account_number not in self.accounts:
            print(f"Error: Account {account_number} not found.")
            return None
        return self.accounts[account_number]
    
    def run(self) -> None:
        print("=" * 50)
        print("Welcome to Interactive Bank Account System")
        print("=" * 50)
        
        while True:
            print("\n--- Main Menu ---")
            print("1. Create new account")
            print("2. View all accounts")
            print("3. Deposit money")
            print("4. Withdraw money")
            print("5. Transfer money")
            print("6. View transaction history")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                self.create_account()
            
            elif choice == '2':
                self.list_accounts()
            
            elif choice == '3':
                account = self.view_account(input("Enter account number: ").strip())
                if account:
                    amount = input("Enter deposit amount: $").strip()
                    if amount:
                        try:
                            self.accounts[account.account_number].deposit(float(amount))
                        except ValueError:
                            print("Error: Invalid amount. Please enter a number.")
            
            elif choice == '4':
                account = self.view_account(input("Enter account number: ").strip())
                if account:
                    amount = input("Enter withdrawal amount: $").strip()
                    if amount:
                        try:
                            account.withdraw(float(amount))
                        except ValueError:
                            print("Error: Invalid amount. Please enter a number.")
            
            elif choice == '5':
                source = self.view_account(input("Enter source account number: ").strip())
                if source:
                    target = self.view_account(input("Enter target account number: ").strip())
                    if target:
                        amount = input("Enter transfer amount: $").strip()
                        if amount:
                            try:
                                source.transfer(target, float(amount))
                            except ValueError:
                                print("Error: Invalid amount. Please enter a number.")
            
            elif choice == '6':
                account = self.view_account(input("Enter account number: ").strip())
                if account:
                    account.view_history()
            
            elif choice == '7':
                print("\nThank you for using the Bank Account System. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-7.")

if __name__ == '__main__':
    bank = BankSystem()
    bank.run()