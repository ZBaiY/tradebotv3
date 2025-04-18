"""Should be directly called by the real-time trading module to allocate capital to crypto."""
"""Not a part of the strategy module."""
#####
# Deactivate for my personal use
#####
# capital_allocator.py
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json


"""
The CapitalAllocator module will allocate a percentage of the total savings to crypto.
Will be directly used in the real-time trading module.s
"""

class CapitalAllocator:
    def __init__(self, equity, config_path='config/capital.json'):
        self.load_config(config_path)
        self.salary = self.config['salary']
        self.saving_interest = self.config['saving_interest']
        self.allocate_method = self.config['allocate_method']
        self.capital_allocation_percentage = self.config.get('capital_allocation_percentage', 0.1)
        self.equity = equity
        self.balances = None

    def set_equity(self, equity):
        self.equity = equity
    
    def set_balances(self, balances):
        self.balances = balances

    def percentage_allocation(self, percentage):
        self.capital_allocation_percentage = percentage
        self.save_config()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)
    
    def calculate_allocated_capital(self):
        """
        Calculates how much capital to allocate to crypto based on salary and savings.
        """
        total_savings = self.config['total_savings']
        allocated_capital = total_savings * self.capital_allocation_percentage
        return allocated_capital

    def constant_allocation(self, amount):
        self.capital_allocation_percentage = amount

    def update_allocation_percentage(self, new_percentage):
        """
        Updates the percentage of savings allocated to crypto.
        """
        self.capital_allocation_percentage = new_percentage
        self.config['capital_allocation_percentage'] = new_percentage
        self.save_config()

    def save_config(self):
        with open('config.json', 'w') as file:
            json.dump(self.config, file)

    def get_allocation_cryp(self):
        self.constant_allocation(1) 
        if self.capital_allocation_percentage == 1:
            return self.equity
        else:
            return self.calculate_allocated_capital()
