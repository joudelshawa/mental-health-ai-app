import matplotlib.pyplot as plt

topics = [ 4000, 1500, 1000,3000, 5000]
colors_religion=['navajowhite','palegreen','turquoise','lightpink','thistle']
plt.pie(topics, colors=colors_religion, startangle=90,frame=True)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# plt.legend(title="indicates", loc="lower right")
# plt.axis('equal')
# plt.tight_layout()
print('about to show!')
# plt.show()
plt.savefig('graph.png', transparent=True)