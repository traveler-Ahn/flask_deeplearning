$(function () {
    var detection_cmd = ['object_detection', 'face_detection',
                         'age_gender_detection', 'emotions_detection', 
                         'facial_landmarks_detection'];
    var flip_cmd = ['flip'];
    var url = "";
    $('.btn').on('click', function () {
        var command = JSON.stringify({ "command": $('#' + $(this).attr('id')).val() });
        if (JSON.parse(command).command == "") {
            var command = JSON.stringify({ "command": $(this).find('input').val() });
        }
        if (detection_cmd.includes(JSON.parse(command).command)) {
            url = '/detection';
            post(url, command);
        }
        if (flip_cmd.includes(JSON.parse(command).command)) {
            url = '/flip';
            post(url, command);
        }
    });
    function post(url, command) {
        $.ajax({
            type: 'POST',
            url: url,
            data: command,
            contentType: 'application/json',
            timeout: 10000
        }).done(function (data) {
            if (data != 'flip') {
                var sent_cmd    = JSON.parse(command).command;
                var is_obj_det  = JSON.parse(data.ResultSet).is_object_detection;
                var is_face_det = JSON.parse(data.ResultSet).is_face_detection;
                var is_ag_det   = JSON.parse(data.ResultSet).is_age_gender_detection;
                var is_em_det   = JSON.parse(data.ResultSet).is_emotions_detection;
                var is_lm_det   = JSON.parse(data.ResultSet).is_facial_landmarks_detection;
                
                $("#res").text("ObjectDetection:" + is_obj_det + "| FaceDetection:" + is_face_det + 
                               "| Age&Gender:" + is_ag_det + "| Emotion:" + is_em_det + "| Landmark:" + is_lm_det);
                if (sent_cmd == 'object_detection') {
                    $("#is_face_detection").attr("disabled", true);
                }
                if (sent_cmd == 'face_detection') {
                    $("#is_face_detection").attr("disabled", false);
                }
            }
        }).fail(function (jqXHR, textStatus, errorThrown) {
            $("#res").text(textStatus + ":" + jqXHR.status + " " + errorThrown);
        });
        return false;
    }
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
});

