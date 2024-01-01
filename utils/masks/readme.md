# mask_collator
```markdown
from Images.utils.masks.maskcollator 
# -- initialize collator 
collator = MaskCollator()

# -- wrap it with dataloader 
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collator) 
```
