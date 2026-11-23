from kaggle_environments import make, evaluate

# 1) Simula a validação do Kaggle: o agente joga contra ele mesmo.
env = make("connectx", debug=True)   # debug=True mostra erros do seu código
env.run(["submission.py", "submission.py"])
print("Status final:", [s.status for s in env.state])  # esperado: DONE e DONE

# 2) Joga contra os adversários embutidos, nos dois lados do tabuleiro.
for oponente in ["random", "negamax"]:
    r1 = evaluate("connectx", ["submission.py", oponente], num_episodes=10)
    r2 = evaluate("connectx", [oponente, "submission.py"], num_episodes=10)
    v = sum(1 for a, b in r1 if a > b) + sum(1 for a, b in r2 if b > a)
    print(f"vs {oponente}: {v}/20 vitórias")