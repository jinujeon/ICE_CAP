# -*- coding: cp949 -*-

"""
 vi:set et ts=4 fenc=cp949:
 Copyright (C) 2008-2010 D&SOFT
 http://open.coolsms.co.kr
"""

import sys
import coolsms


def main():
    # ��ü ����
    cs = coolsms.sms(True)

    # ���α׷���� ������ �Է��մϴ�. (��������)
    cs.appversion("TEST/1.0")

    # �ѱ����ڵ� ����� �����մϴ�.  (������ euckr�� ����)
    # ���� ���ڵ�: euckr, utf8
    cs.charset("euckr")

    # ���̵�� �н����带 �Է��մϴ�.
    # �𿡽����������� ȸ�����Խ� �Է��� ���̵�/��й�ȣ�� �Է��մϴ�.
    cs.setuser("cs_id", "cs_passwd")

    # MMS ÷�� �̹����� �ִ� ��� (���ڸ� �������� ������ ���� ���丮�� ���� ��)
    cs.setattachdir("")

    # add("�޴»�� ����ȣ", "�����»�� ����ȣ", "MMS���� (20 bytes)",
    # "MMS �޽��� ����(2000 bytes)", "�̹������ϸ� (200 Kbytes)")
    # "�������۽ð�"�� ���� �ϰų� ���� �ð����� �����ð����� �����ϸ� ��� ���� ��
    # �������� ǥ��� : YYYYMMDDhhmmss (YYYY=��, MM=��, DD=��, hh=��, mm=��, ss=��)
    # String ���� ǥ���ϸ� ss(��)�� ���� ����

    # ��� ���۽�
    cs.addsms("01012341234", "0212341234", "MMS���� 20����Ʈ����", "2000����Ʈ���� �ؽ�Ʈ�� ������ �� �ֽ��ϴ�.", "test.jpg")

    # ���� ���۽�
    cs.addsms("01012341234", "0212341234", "MMS���� 20����Ʈ����", "2000����Ʈ���� �ؽ�Ʈ�� ������ �� �ֽ��ϴ�.", "test.jpg", "YYYYMMDDhhmm")

    # cs.addmms �޼ҵ带 ��� ȣ���Ͽ� �޽����� �߰� �� �� ����.
    
    nsent = 0
    if cs.connect():
        # add �� ��� �޼����� ������ �����ϴ�.
        nsent = cs.send()
    else:
        # ����ó��
        print "������ ������ �� �����ϴ�. ��Ʈ��ũ ���¸� Ȯ���ϼ���."

    # ���� ����
    cs.disconnect()

    # ����� ����մϴ�.
    print "%d ���� ������ ����Դϴ�." % nsent
    cs.printr()

    # �޸� �ʱ�ȭ
    cs.emptyall()


if __name__ == "__main__":
    main()
    sys.exit(0)
