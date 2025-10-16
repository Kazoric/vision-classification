from tensorboard import program

logdir = "runs"  # Change si besoin

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir])
url = tb.launch()
print(f"🎉 TensorBoard lancé sur : {url}")
