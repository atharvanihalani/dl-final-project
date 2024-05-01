# File for global variables

TRAIN_IN_SIZE = () # should be (batch, timesteps, frequency bins)
TRAIN_OUT_SIZE = () # should be (batch, timesteps (different scale from above), 88)

TEST_IN_SIZE = ()
TEST_OUT_SIZE = ()

# bach 1 shape: (32, 110250, 88)

SAMPLING_RATE = 22050
SECONDS = 5
BINS_PER_OCTAVE = 36
START_TOKENS = 500
END_TOKENS = -500
