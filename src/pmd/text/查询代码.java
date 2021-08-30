 for i in range(len(list1)):
 'account_ids': fields.many2many('account.account', 'reconcile_account_rel', 'reconcile_id', 'account_id', 'Accounts to Reconcile', domain = [('reconcile','=',1)],),
