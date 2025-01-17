import numpy as np

def get_distance(vid_num, start_pos, end_pos):
	"""
	Calculate distance travelled by player per frame

	*Note:
	- Dimension of actual court: 16x8 (metres)
	- Dimension of top-down court: 480x240 (pixels)

	Args:
		start_pos
	"""
	m_per_pixel = 0.0333
	if vid_num == 1:
		m_per_pixel = 0.02
	if vid_num == 4:
		m_per_pixel = 0.033
	distance = np.linalg.norm(start_pos - end_pos) * m_per_pixel

	return distance