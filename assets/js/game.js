const X = 'x';
const Y = 'y';
const SPEED_X = 'sx';
const SPEED_Y = 'sy';
const WIDTH = 'w';
const HEIGHT = 'h';
const WALL_TAG_TOP = 'wt';
const WALL_TAG_BOTTOM = 'wb';
const LIVING = 'l';
const PLAYER_TAG = 'player';
const WALLS_TAG = 'walls';
const SCORE_TAG = 'score';
function newObj(x, y, width, height, speedX, speedY) {
    var ret = {};
    try {
        ret[X] = Number(x);
        ret[Y] = Number(y);
        ret[SPEED_X] = Number(speedX);
        ret[SPEED_Y] = Number(speedY);
        ret[WIDTH] = Number(width);
        ret[HEIGHT] = Number(height);
        ret[LIVING] = true;
    } catch (error) {
        alert(error)
    }
    return ret;
}
function clone(obj) {
    if (Object.prototype.toString.call(obj) == '[object Object]') {
        let ret = {}
        var keys = Object.keys(obj);
        for (let i = 0, len = keys.length; i < len; i++) {
            ret[keys[i]] = clone(obj[keys[i]]);
        }
        return ret;
    } else if (Object.prototype.toString.call(obj) == '[object Array]') {
        let ret = []
        for (let i = 0, len = obj.length; i < len; i++) {
            ret.push(clone(obj[i]));
        }
        return ret;
    } else
        return obj;
}
function newGame(screenWidth, screenHeight, playerX, playerY, playerWidth, playerHeight, wallWidth, doorHeight,
    wallDelay, minWallDelay, difficut, wallSpeedX, eventEffect, g) {
    const DOOR_HEIGHT = doorHeight;
    const MIN_WALL_DELAY = minWallDelay;
    const DIFFICULT = difficut;
    const WALL_WIDTH = wallWidth;
    const SCREEN_WIDTH = screenWidth;
    const SCREEN_HEIGHT = screenHeight;
    const PLAYER_WIDTH = playerWidth;
    const PLAYER_HEIGHT = playerHeight;
    const QUEUE_SIZE = 300;
    const WALL_SPEED = wallSpeedX;
    const EFFECT = eventEffect;
    const G = g;
    var screen = newObj(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0);
    var player = newObj(playerX, playerY, PLAYER_WIDTH, PLAYER_HEIGHT, 0, 0);
    var wallQueue = [];
    var wallList = [];
    var saves = [];
    var score = 0;
    var wallCooldown = Math.abs(wallDelay - 1);
    var wallMaxCooldown = wallDelay;
    var nextWall = function () {
        while ((!wallQueue) || wallQueue.length <= 0) {
            wallQueue = []
            for (let i = 0; i < QUEUE_SIZE; i++) {
                var doorTop = Math.random() * (SCREEN_HEIGHT * 0.8 - DOOR_HEIGHT) + SCREEN_HEIGHT * 0.1;
                var topWall = newObj(SCREEN_WIDTH + 1, 0, WALL_WIDTH, doorTop, WALL_SPEED, 0);
                var bottomWall = newObj(SCREEN_WIDTH + 1, doorTop + DOOR_HEIGHT, WALL_WIDTH, SCREEN_HEIGHT - (doorTop + DOOR_HEIGHT), WALL_SPEED, 0);
                var res = {};
                res[WALL_TAG_TOP] = topWall;
                res[WALL_TAG_BOTTOM] = bottomWall;
                wallQueue.push(res);
            }
        }
        return wallQueue.shift();
    }
    var appendWall = function () {
        var res = nextWall();
        wallList.push(res[WALL_TAG_BOTTOM]);
        wallList.push(res[WALL_TAG_TOP]);
    }
    var checkCollision = function (obj0, obj1) {
        let ret = true;
        if (obj0[X] > obj1[X] + obj1[WIDTH] || obj0[X] + obj0[WIDTH] < obj1[X] ||
            obj0[Y] > obj1[Y] + obj1[HEIGHT] || obj0[Y] + obj0[HEIGHT] < obj1[Y]) {
            ret = false;
        }
        return ret;
    }
    var checkLiving = function () {
        if (!checkCollision(screen, player))
            return false;
        for (let i = 0, len = wallList.length; i < len; i++) {
            if (checkCollision(player, wallList[i])) {
                return false;
            }
        }
        return true;
    }
    var checkGoal = function () {
        let ret = false;
        for (let i = 0, len = wallList.length; i < len; i++) {
            if (player[X] > wallList[i][X] + wallList[i][WIDTH] && wallList[i][LIVING]) {
                wallList[i][LIVING] = false;
                ret = true;
            }
        }
        return ret;
    }
    var wallClean = function () {
        for (let i = wallList.length - 1; i >= 0; i--) {
            if (screen[X] > wallList[i][X] + wallList[i][WIDTH]) {
                wallList.splice(i, 1);
            }
        }
    }
    var move = function (event) {
        if (event) {
            player[SPEED_Y] = EFFECT;
        } else {
            player[SPEED_Y] = player[SPEED_Y] + G;
        }
        player[Y] += player[SPEED_Y];
        for (let i = 0, len = wallList.length; i < len; i++) {
            wallList[i][X] -= wallList[i][SPEED_X];
        }
    }
    var save = function () {
        var now = {};
        now[PLAYER_TAG] = clone(player);
        now[WALLS_TAG] = clone(wallList);
        now[SCORE_TAG] = score;
        saves.push(now);
        return now;
    }
    var run = function (event) {
        if (player[LIVING]) {
            move(event);
            wallClean();
            wallCooldown = (wallCooldown + 1) % wallMaxCooldown;
            if (wallCooldown == 0) {
                appendWall();
                wallMaxCooldown = Math.max(wallMaxCooldown - DIFFICULT, MIN_WALL_DELAY);
            }
            if (checkGoal()) {
                score++;
            }
            player[LIVING] = checkLiving();
        }
        return save();
    };
    var next = function () {
        if (saves.length <= 0)
            run(false);
        return saves.shift();
    }
    return {
        run: run,
        next: next,
        getScore: function () {
            return score;
        },
        getSize: function () {
            return saves.length;
        },
        peek: function () {
            return saves[saves.length - 1];
        }
    }
}