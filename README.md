
# Fincrime Triage Agent

A small-scale model of a bank's financial-crime operations queue: sanctions screening, transaction monitoring, and an AI agent that triages flagged cases, auto-closing routine ones and escalating the rest to a human.

## Status
In progress: Stage 1 (loading and normalizing the OFAC SDN list)

## Roadmap
- [ ] Stage 1: Load and clean the OFAC sanctions list
- [ ] Stage 2: Generate synthetic customers and transactions
- [ ] Stage 3: Name screening (exact → fuzzy matching)
- [ ] Stage 4: Transaction monitoring rules
- [ ] Stage 5: AI case reviewer
- [ ] Stage 6: Results report

## Data
Sanctions data comes from the U.S. Treasury's OFAC SDN list (public). Download `SDN.CSV` and `ALT.CSV` into `data/`. All customer and transaction data is synthetic.

See [LEARNING_LOG.md](LEARNING_LOG.md) for what I'm learning along the way.
